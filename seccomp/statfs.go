//
// Copyright 2024 Nestybox, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package seccomp

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/nestybox/sysbox-fs/btrfsutil"
	"github.com/nestybox/sysbox-fs/domain"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// processStatfs handles intercepted statfs/statfs64 syscalls.
// statfs(const char *path, struct statfs *buf)
//
//	Args[0] = path pointer
//	Args[1] = buf pointer
//
// Implementation:
//
// We resolve the tracee-visible path on the host by composing it with the
// tracee's procfs entry: /proc/<pid>/root/<absPath> for absolute paths, or
// /proc/<pid>/cwd/<relPath> for relative paths. Both /proc/<pid>/root and
// /proc/<pid>/cwd are magic symlinks that the kernel follows in the
// tracee's mount namespace, so any container-only mount (bind mounts,
// overlays, etc) is resolved correctly without spawning an nsenter helper.
//
// This avoids the per-syscall fork+setns cost that would otherwise dominate
// workloads doing many statfs(2) calls in a row (df, mount, etc).
func (t *syscallTracer) processStatfs(
	req *sysRequest,
	fd int32,
	cntr domain.ContainerIface,
	syscallName string) (*sysResponse, error) {

	// Read the "path" argument from the tracee's memory.
	parsedArgs, err := t.memParser.ReadSyscallStringArgs(
		req.Pid,
		[]memParserDataElem{{req.Data.Args[0], unix.PathMax, nil}},
	)
	if err != nil {
		return t.createContinueResponse(req.ID), nil
	}
	path := parsedArgs[0]
	bufAddr := req.Data.Args[1]

	if path == "" {
		return t.createContinueResponse(req.ID), nil
	}

	// Compose the host-visible path that resolves in the tracee's mount ns.
	hostPath := composeTraceePath(req.Pid, path)

	hostFd, err := btrfsutil.OpenForStatfs(hostPath)
	if err != nil {
		// Path doesn't exist (or can't be opened); let the kernel handle
		// the syscall and return the proper errno to the tracee.
		logrus.Debugf("statfs[%s]: open %s failed: %v - continuing",
			syscallName, hostPath, err)
		return t.createContinueResponse(req.ID), nil
	}
	defer unix.Close(hostFd)

	resp, err := btrfsutil.BuildStatfsResp(uintptr(hostFd),
		fmt.Sprintf("path=%q pid=%d", path, req.Pid))
	if err != nil {
		logrus.Debugf("statfs[%s]: BuildStatfsResp failed: %v - continuing",
			syscallName, err)
		return t.createContinueResponse(req.ID), nil
	}

	if !resp.IsBtrfs || !resp.HasQuota {
		// Not btrfs, or no quota set: nothing to override; let the
		// kernel produce the answer to keep the syscall transparent.
		logrus.Debugf("statfs[%s]: pid=%d path=%q isBtrfs=%v hasQuota=%v - continuing (debug=%q)",
			syscallName, req.Pid, path, resp.IsBtrfs, resp.HasQuota, resp.Debug)
		return t.createContinueResponse(req.ID), nil
	}

	logrus.Infof("statfs[%s]: pid=%d path=%q quota max=%d used=%d "+
		"blocks=%d bavail=%d bsize=%d (subvolId=%d)",
		syscallName, req.Pid, path, resp.MaxReferenced, resp.Referenced,
		resp.Blocks, resp.Bavail, resp.BlockSize, resp.SubvolId)

	statfsBuf := serializeStatfsResp(&resp)
	if err := t.memParser.WriteSyscallBytesArgs(
		req.Pid,
		[]memParserDataElem{{bufAddr, len(statfsBuf), statfsBuf}},
	); err != nil {
		logrus.Warnf("statfs[%s]: failed to write statfs result to tracee memory: %v",
			syscallName, err)
		return t.createContinueResponse(req.ID), nil
	}

	return t.createSuccessResponse(req.ID), nil
}

// processFstatfs handles intercepted fstatfs/fstatfs64 syscalls.
// fstatfs(int fd, struct statfs *buf)
//
//	Args[0] = fd (in tracee's fd table)
//	Args[1] = buf pointer (in tracee's address space)
//
// Unlike a hypothetical "fd-to-path" approach we do NOT readlink the fd back
// to a path: that would diverge from fstatfs(2)'s kernel semantics (which
// binds to the file the fd was opened against, including unlinked files,
// files on now-unmounted vfsmounts, magic links, etc).
//
// Instead, the tracer (running on the host as root) reopens the tracee's fd
// directly via /proc/<host-pid>/fd/<fd>. The kernel resolves this magic
// symlink to the same struct file the tracee has, giving us a fd that
// references the very same inode + vfsmount. fstatfs(2) on this fd produces
// the exact result the tracee would have seen.
//
// The btrfs ioctls we use to look up the subvolume id and qgroup limits
// only depend on the file's inode and its containing superblock, both of
// which are namespace-independent, so we can run them on the host without
// entering any container namespaces.
func (t *syscallTracer) processFstatfs(
	req *sysRequest,
	fd int32,
	cntr domain.ContainerIface,
	syscallName string) (*sysResponse, error) {

	traceeFd := int32(req.Data.Args[0])
	bufAddr := req.Data.Args[1]

	procPath := fmt.Sprintf("/proc/%d/fd/%d", req.Pid, traceeFd)
	hostFd, err := btrfsutil.OpenForStatfs(procPath)
	if err != nil {
		// Either the tracee fd is gone (raced) or magic-link reopen
		// is not allowed for this fd kind. Let the kernel handle the
		// syscall - it will produce the right answer or the right error.
		logrus.Debugf("fstatfs[%s]: reopen %s failed: %v - continuing",
			syscallName, procPath, err)
		return t.createContinueResponse(req.ID), nil
	}
	defer unix.Close(hostFd)

	resp, err := btrfsutil.BuildStatfsResp(uintptr(hostFd),
		fmt.Sprintf("fd=%d pid=%d", traceeFd, req.Pid))
	if err != nil {
		logrus.Debugf("fstatfs[%s]: BuildStatfsResp failed: %v - continuing",
			syscallName, err)
		return t.createContinueResponse(req.ID), nil
	}

	if !resp.IsBtrfs || !resp.HasQuota {
		logrus.Debugf("fstatfs[%s]: pid=%d fd=%d isBtrfs=%v hasQuota=%v - continuing (debug=%q)",
			syscallName, req.Pid, traceeFd, resp.IsBtrfs, resp.HasQuota, resp.Debug)
		return t.createContinueResponse(req.ID), nil
	}

	logrus.Infof("fstatfs[%s]: pid=%d fd=%d quota max=%d used=%d "+
		"blocks=%d bavail=%d bsize=%d (subvolId=%d)",
		syscallName, req.Pid, traceeFd, resp.MaxReferenced, resp.Referenced,
		resp.Blocks, resp.Bavail, resp.BlockSize, resp.SubvolId)

	statfsBuf := serializeStatfsResp(&resp)
	if err := t.memParser.WriteSyscallBytesArgs(
		req.Pid,
		[]memParserDataElem{{bufAddr, len(statfsBuf), statfsBuf}},
	); err != nil {
		logrus.Warnf("fstatfs[%s]: failed to write statfs result to tracee memory: %v",
			syscallName, err)
		return t.createContinueResponse(req.ID), nil
	}

	return t.createSuccessResponse(req.ID), nil
}

// composeTraceePath returns a host path that, when opened by the host, is
// resolved by the kernel inside the tracee's mount namespace.
//
//   - Absolute paths use /proc/<pid>/root as the lookup root: the magic
//     symlink "root" pins lookup to the tracee's fs_struct->root, including
//     its mount table.
//   - Relative paths use /proc/<pid>/cwd, similarly pinned to the tracee's
//     working directory in its own mount ns.
//
// We strip a leading "./" and collapse internal "/." components defensively;
// the kernel itself does this during path lookup, so we mostly do it for
// cleaner debug output.
//
// Note: the resulting host path may contain ".." components from the tracee.
// That's safe: when path lookup follows the magic symlink prefix, the kernel
// treats it as the lookup root and ".." cannot escape it (same behaviour as
// chroot).
func composeTraceePath(pid uint32, path string) string {
	cleaned := filepath.Clean(path)

	if strings.HasPrefix(cleaned, "/") {
		return fmt.Sprintf("/proc/%d/root%s", pid, cleaned)
	}
	return fmt.Sprintf("/proc/%d/cwd/%s", pid, cleaned)
}

// serializeStatfsResp serializes a btrfsutil.StatfsResult into the on-the-wire
// layout of struct statfs (as the kernel would write it back).
//
// On amd64/arm64, struct statfs (as defined in <sys/statfs.h>) has the
// following layout (size = 120 bytes):
//
//	__fsword_t f_type;       (8 bytes, offset 0)
//	__fsword_t f_bsize;      (8 bytes, offset 8)
//	fsblkcnt_t f_blocks;     (8 bytes, offset 16)
//	fsblkcnt_t f_bfree;      (8 bytes, offset 24)
//	fsblkcnt_t f_bavail;     (8 bytes, offset 32)
//	fsfilcnt_t f_files;      (8 bytes, offset 40)
//	fsfilcnt_t f_ffree;      (8 bytes, offset 48)
//	__fsid_t   f_fsid;       (8 bytes, offset 56)
//	__fsword_t f_namelen;    (8 bytes, offset 64)
//	__fsword_t f_frsize;     (8 bytes, offset 72)
//	__fsword_t f_flags;      (8 bytes, offset 80)
//	__fsword_t f_spare[4];   (32 bytes, offset 88)
//
// statfs64 has the same layout on these architectures.
func serializeStatfsResp(r *btrfsutil.StatfsResult) []byte {
	var stfs unix.Statfs_t
	size := int(unsafe.Sizeof(stfs))
	buf := make([]byte, size)

	binary.LittleEndian.PutUint64(buf[0:8], uint64(r.StatfsType))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(r.BlockSize))
	binary.LittleEndian.PutUint64(buf[16:24], r.Blocks)
	binary.LittleEndian.PutUint64(buf[24:32], r.Bfree)
	binary.LittleEndian.PutUint64(buf[32:40], r.Bavail)
	binary.LittleEndian.PutUint64(buf[40:48], r.Files)
	binary.LittleEndian.PutUint64(buf[48:56], r.Ffree)
	binary.LittleEndian.PutUint32(buf[56:60], uint32(r.FsidVal0))
	binary.LittleEndian.PutUint32(buf[60:64], uint32(r.FsidVal1))
	binary.LittleEndian.PutUint64(buf[64:72], uint64(r.Namelen))
	binary.LittleEndian.PutUint64(buf[72:80], uint64(r.Frsize))
	binary.LittleEndian.PutUint64(buf[80:88], uint64(r.Flags))
	// f_spare[4] left zero.
	return buf
}
