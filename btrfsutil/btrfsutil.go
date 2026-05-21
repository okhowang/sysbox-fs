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

// Package btrfsutil provides helpers to query btrfs subvolume id and qgroup
// quota info for a given file descriptor.
//
// All operations are fd-based on purpose: this preserves the kernel semantics
// of fstatfs(2) (a query against the file/inode the fd refers to, not against
// a path that has to be re-resolved). It also lets callers run from any mount
// namespace, since btrfs ioctls operate on the superblock backing the fd.
package btrfsutil

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"golang.org/x/sys/unix"
)

// btrfs ioctl commands
//
// Encoding (Linux _IOWR macro, identical on amd64/arm64/...):
//
//	(_IOC_READ|_IOC_WRITE)<<30 | size<<16 | type<<8 | nr
//	= (3 << 30) | (4096 << 16) | (0x94 << 8) | nr
//	= 0xD0009400 | nr
//
// Both BTRFS_IOC_INO_LOOKUP (nr=18) and BTRFS_IOC_TREE_SEARCH (nr=17) take
// argument structs whose total size is exactly 4096 bytes.
const (
	// BTRFS_IOC_INO_LOOKUP: _IOWR(BTRFS_IOCTL_MAGIC, 18, struct btrfs_ioctl_ino_lookup_args)
	btrfsIocInoLookup = 0xD0009412

	// BTRFS_IOC_TREE_SEARCH: _IOWR(BTRFS_IOCTL_MAGIC, 17, struct btrfs_ioctl_search_args)
	btrfsIocTreeSearch = 0xD0009411
)

// btrfs tree search key types
const (
	btrfsQgroupInfoKey  = 242
	btrfsQgroupLimitKey = 244
)

// btrfs tree object ids
const (
	// BTRFS_QUOTA_TREE_OBJECTID: tree id of the quota tree
	btrfsQuotaTreeObjectid = 8
	// BTRFS_FIRST_FREE_OBJECTID: first inode objectid for files in a subvolume
	btrfsFirstFreeObjectid = 256
)

// SuperMagic is the magic number for btrfs filesystem (matches stfs.Type).
const SuperMagic = 0x9123683E

// btrfsIoctlInoLookupArgs corresponds to struct btrfs_ioctl_ino_lookup_args.
// Total size: 8 + 8 + 4080 = 4096 bytes (matches kernel exactly on all archs).
type btrfsIoctlInoLookupArgs struct {
	Treeid   uint64
	Objectid uint64
	Name     [4080]byte
}

// btrfsIoctlSearchKey corresponds to struct btrfs_ioctl_search_key (104 bytes).
type btrfsIoctlSearchKey struct {
	TreeId      uint64
	MinObjectid uint64
	MaxObjectid uint64
	MinOffset   uint64
	MaxOffset   uint64
	MinTransid  uint64
	MaxTransid  uint64
	MinType     uint32
	MaxType     uint32
	NrItems     uint32
	_unused     uint32
	_unused1    uint64
	_unused2    uint64
	_unused3    uint64
	_unused4    uint64
}

// btrfsIoctlSearchHeader corresponds to struct btrfs_ioctl_search_header.
type btrfsIoctlSearchHeader struct {
	Transid  uint64
	Objectid uint64
	Offset   uint64
	Type     uint32
	Len      uint32
}

// btrfsIoctlSearchArgs corresponds to struct btrfs_ioctl_search_args.
// Total size = sizeof(search_key) + BTRFS_SEARCH_ARGS_BUFSIZE = 4096.
const btrfsSearchArgsBufSize = 4096 - 104

type btrfsIoctlSearchArgs struct {
	Key btrfsIoctlSearchKey
	Buf [btrfsSearchArgsBufSize]byte
}

// btrfsQgroupLimitItem corresponds to struct btrfs_qgroup_limit_item.
type btrfsQgroupLimitItem struct {
	Flags         uint64
	MaxReferenced uint64
	MaxExclusive  uint64
	RsvReferenced uint64
	RsvExclusive  uint64
}

// btrfsQgroupInfoItem corresponds to struct btrfs_qgroup_info_item.
type btrfsQgroupInfoItem struct {
	Generation          uint64
	Referenced          uint64
	ReferredCompressed  uint64
	Exclusive           uint64
	ExclusiveCompressed uint64
}

// QuotaInfo holds qgroup quota information for a btrfs subvolume.
type QuotaInfo struct {
	HasQuota      bool   // whether a non-zero max_rfer is set on the qgroup
	MaxReferenced uint64 // max bytes allowed (0 means no limit)
	Referenced    uint64 // current bytes used
}

// StatfsResult is the combined output of fstatfs(2) plus optional btrfs
// qgroup quota lookup, as produced by BuildStatfsResp. Fields prefixed with
// "Statfs" come from the kernel's struct statfs as-is; the remaining fields
// are filled only when the underlying filesystem is btrfs and a qgroup is
// configured for the containing subvolume.
type StatfsResult struct {
	// IsBtrfs is true when stfs.Type == SuperMagic.
	IsBtrfs bool

	// HasQuota is true when a non-zero qgroup max_rfer is in effect for
	// the subvolume containing the inode. When true, Blocks/Bfree/Bavail
	// have been rewritten to reflect the quota (in BlockSize units) and
	// the caller is expected to forward this view to its tracee.
	HasQuota bool

	// Statfs fields, copied (and possibly rewritten when HasQuota) from
	// the kernel's struct statfs.
	StatfsType int64
	BlockSize  int64
	Blocks     uint64
	Bfree      uint64
	Bavail     uint64
	Files      uint64
	Ffree      uint64
	Namelen    int64
	Frsize     int64
	Flags      int64
	FsidVal0   int32
	FsidVal1   int32

	// Btrfs-specific diagnostics, populated only when IsBtrfs.
	SubvolId      uint64
	MaxReferenced uint64
	Referenced    uint64

	// Debug is a free-form trace of the lookup steps; useful when an
	// expected quota is not visible (e.g. ioctl errors, qgroup absent).
	Debug string
}

// OpenForStatfs opens path with the right flags for the rest of this package
// to work, namely:
//
//   - try O_RDONLY|O_NONBLOCK|O_NOFOLLOW first, since the btrfs ioctls used
//     by GetSubvolumeId / GetQuotaInfo refuse O_PATH fds (EBADF);
//   - if that fails, try O_RDONLY without O_NOFOLLOW (statfs(2) follows the
//     final symlink);
//   - if that still fails (typical for sockets / pipes / anon inodes that
//     can't be reopened O_RDONLY), fall back to O_PATH so at least fstatfs(2)
//     keeps working - quota lookup will be skipped.
//
// O_NONBLOCK is set so that opening a FIFO does not block waiting for a peer.
//
// Returns a fd suitable for use with BuildStatfsResp. Caller must Close it.
func OpenForStatfs(path string) (int, error) {
	flags := unix.O_RDONLY | unix.O_NONBLOCK | unix.O_CLOEXEC | unix.O_NOFOLLOW
	if fd, err := unix.Open(path, flags, 0); err == nil {
		return fd, nil
	}
	flags = unix.O_RDONLY | unix.O_NONBLOCK | unix.O_CLOEXEC
	if fd, err := unix.Open(path, flags, 0); err == nil {
		return fd, nil
	}
	flags = unix.O_PATH | unix.O_CLOEXEC
	return unix.Open(path, flags, 0)
}

// GetSubvolumeId returns the subvolume id (i.e. the tree id of the subvolume
// the fd lives in). Caller retains ownership of fd.
//
// Implementation: BTRFS_IOC_INO_LOOKUP with treeid=0 (=> the fd's own subvol)
// and objectid=BTRFS_FIRST_FREE_OBJECTID (256, the inode that any subvol root
// directory has). The kernel fills args.Treeid with the resolved subvol id.
func GetSubvolumeId(fd uintptr) (uint64, error) {
	var args btrfsIoctlInoLookupArgs
	args.Treeid = 0
	args.Objectid = btrfsFirstFreeObjectid

	_, _, errno := unix.Syscall(unix.SYS_IOCTL, fd,
		uintptr(btrfsIocInoLookup),
		uintptr(unsafe.Pointer(&args)))
	if errno != 0 {
		return 0, fmt.Errorf("BTRFS_IOC_INO_LOOKUP failed: %w", errno)
	}
	return args.Treeid, nil
}

// GetQuotaInfo queries the qgroup limit + info for the given subvolume id,
// using fd as a handle into the btrfs filesystem (any fd in the same fs works).
//
// If qgroups are disabled or no row exists for this subvolume, returns a
// zero-value QuotaInfo with HasQuota=false (and no error).
func GetQuotaInfo(fd uintptr, subvolId uint64) (*QuotaInfo, error) {
	info := &QuotaInfo{}

	limit, err := queryQgroupLimit(fd, subvolId)
	if err != nil {
		// qgroup tree absent => quotas disabled. Not a hard error.
		return info, nil
	}
	if limit != nil && limit.MaxReferenced > 0 {
		info.HasQuota = true
		info.MaxReferenced = limit.MaxReferenced
	}

	if qinfo, err := queryQgroupInfo(fd, subvolId); err == nil && qinfo != nil {
		info.Referenced = qinfo.Referenced
	}

	return info, nil
}

// queryQgroupLimit performs a tree search on the quota tree for the
// BTRFS_QGROUP_LIMIT_KEY of qgroup 0/<subvolId>.
func queryQgroupLimit(fd uintptr, subvolId uint64) (*btrfsQgroupLimitItem, error) {
	args, err := treeSearchSingle(fd, subvolId, btrfsQgroupLimitKey)
	if err != nil || args == nil {
		return nil, err
	}

	header := (*btrfsIoctlSearchHeader)(unsafe.Pointer(&args.Buf[0]))
	if header.Type != btrfsQgroupLimitKey {
		return nil, nil
	}

	dataOff := unsafe.Sizeof(btrfsIoctlSearchHeader{})
	if int(dataOff)+int(header.Len) > len(args.Buf) {
		return nil, fmt.Errorf("search result buffer overflow")
	}
	itemData := args.Buf[dataOff : int(dataOff)+int(header.Len)]
	if len(itemData) < int(unsafe.Sizeof(btrfsQgroupLimitItem{})) {
		return nil, fmt.Errorf("insufficient data for qgroup limit item")
	}

	return &btrfsQgroupLimitItem{
		Flags:         binary.LittleEndian.Uint64(itemData[0:8]),
		MaxReferenced: binary.LittleEndian.Uint64(itemData[8:16]),
		MaxExclusive:  binary.LittleEndian.Uint64(itemData[16:24]),
		RsvReferenced: binary.LittleEndian.Uint64(itemData[24:32]),
		RsvExclusive:  binary.LittleEndian.Uint64(itemData[32:40]),
	}, nil
}

// queryQgroupInfo performs a tree search on the quota tree for the
// BTRFS_QGROUP_INFO_KEY of qgroup 0/<subvolId>.
func queryQgroupInfo(fd uintptr, subvolId uint64) (*btrfsQgroupInfoItem, error) {
	args, err := treeSearchSingle(fd, subvolId, btrfsQgroupInfoKey)
	if err != nil || args == nil {
		return nil, err
	}

	header := (*btrfsIoctlSearchHeader)(unsafe.Pointer(&args.Buf[0]))
	if header.Type != btrfsQgroupInfoKey {
		return nil, nil
	}

	dataOff := unsafe.Sizeof(btrfsIoctlSearchHeader{})
	if int(dataOff)+int(header.Len) > len(args.Buf) {
		return nil, fmt.Errorf("search result buffer overflow")
	}
	itemData := args.Buf[dataOff : int(dataOff)+int(header.Len)]
	if len(itemData) < int(unsafe.Sizeof(btrfsQgroupInfoItem{})) {
		return nil, fmt.Errorf("insufficient data for qgroup info item")
	}

	return &btrfsQgroupInfoItem{
		Generation:          binary.LittleEndian.Uint64(itemData[0:8]),
		Referenced:          binary.LittleEndian.Uint64(itemData[8:16]),
		ReferredCompressed:  binary.LittleEndian.Uint64(itemData[16:24]),
		Exclusive:           binary.LittleEndian.Uint64(itemData[24:32]),
		ExclusiveCompressed: binary.LittleEndian.Uint64(itemData[32:40]),
	}, nil
}

// treeSearchSingle issues BTRFS_IOC_TREE_SEARCH for one specific key in the
// quota tree (objectid=0, offset=subvolId, type=keyType). Returns nil args
// when the kernel found no matching item.
func treeSearchSingle(fd uintptr, subvolId uint64, keyType uint32) (*btrfsIoctlSearchArgs, error) {
	var args btrfsIoctlSearchArgs

	args.Key.TreeId = btrfsQuotaTreeObjectid
	args.Key.MinObjectid = 0
	args.Key.MaxObjectid = 0
	args.Key.MinOffset = subvolId
	args.Key.MaxOffset = subvolId
	args.Key.MinType = keyType
	args.Key.MaxType = keyType
	args.Key.MinTransid = 0
	args.Key.MaxTransid = ^uint64(0)
	args.Key.NrItems = 1

	_, _, errno := unix.Syscall(unix.SYS_IOCTL, fd,
		uintptr(btrfsIocTreeSearch),
		uintptr(unsafe.Pointer(&args)))
	if errno != 0 {
		return nil, fmt.Errorf("BTRFS_IOC_TREE_SEARCH failed: %w", errno)
	}
	if args.Key.NrItems == 0 {
		return nil, nil
	}
	return &args, nil
}

// BuildStatfsResp performs fstatfs(2) on the given fd and, if the underlying
// filesystem is btrfs, queries qgroup quota for the subvolume containing the
// fd's inode. It returns a fully populated StatfsResult, including a Debug
// string useful for log forensics. Caller retains ownership of fd.
//
// Important: fstatfs(2) accepts an O_PATH fd, but the btrfs ioctls
// (BTRFS_IOC_INO_LOOKUP, BTRFS_IOC_TREE_SEARCH) reject O_PATH with EBADF.
// To get quota information, the fd must be a "real" open (e.g. O_RDONLY).
// If only an O_PATH fd is available, IsBtrfs will still be set correctly,
// but HasQuota will be false (with subvolId-err recorded in Debug).
//
// debugTag is included in resp.Debug to help correlate log lines with the
// original syscall (e.g. "fd=12 path=...").
func BuildStatfsResp(fd uintptr, debugTag string) (StatfsResult, error) {
	var stfs unix.Statfs_t
	if err := unix.Fstatfs(int(fd), &stfs); err != nil {
		return StatfsResult{}, err
	}

	resp := StatfsResult{
		IsBtrfs:    stfs.Type == SuperMagic,
		StatfsType: int64(stfs.Type),
		BlockSize:  int64(stfs.Bsize),
		Blocks:     stfs.Blocks,
		Bfree:      stfs.Bfree,
		Bavail:     stfs.Bavail,
		Files:      stfs.Files,
		Ffree:      stfs.Ffree,
		Namelen:    int64(stfs.Namelen),
		Frsize:     int64(stfs.Frsize),
		Flags:      int64(stfs.Flags),
		FsidVal0:   stfs.Fsid.Val[0],
		FsidVal1:   stfs.Fsid.Val[1],
	}
	resp.Debug = fmt.Sprintf("%s type=0x%x", debugTag, stfs.Type)

	if !resp.IsBtrfs {
		resp.Debug += " not-btrfs"
		return resp, nil
	}

	subvolId, err := GetSubvolumeId(fd)
	if err != nil {
		resp.Debug += fmt.Sprintf(" subvolId-err=%v", err)
		return resp, nil
	}
	resp.SubvolId = subvolId
	resp.Debug += fmt.Sprintf(" subvolId=%d", subvolId)

	quotaInfo, err := GetQuotaInfo(fd, subvolId)
	if err != nil {
		resp.Debug += fmt.Sprintf(" quotaInfo-err=%v", err)
		return resp, nil
	}
	if quotaInfo == nil {
		resp.Debug += " quotaInfo=nil"
		return resp, nil
	}
	resp.MaxReferenced = quotaInfo.MaxReferenced
	resp.Referenced = quotaInfo.Referenced
	resp.Debug += fmt.Sprintf(" max=%d used=%d hasQuota=%v",
		quotaInfo.MaxReferenced, quotaInfo.Referenced, quotaInfo.HasQuota)

	if !quotaInfo.HasQuota {
		return resp, nil
	}

	// Quota present: rewrite blocks/bfree/bavail to reflect the quota,
	// in units of stfs.Bsize. If Bsize is 0 we leave the values untouched
	// (this should never happen for a real fs).
	if stfs.Bsize > 0 {
		blockSize := uint64(stfs.Bsize)
		totalBlocks := quotaInfo.MaxReferenced / blockSize
		usedBlocks := quotaInfo.Referenced / blockSize
		var freeBlocks uint64
		if totalBlocks > usedBlocks {
			freeBlocks = totalBlocks - usedBlocks
		}
		resp.Blocks = totalBlocks
		resp.Bfree = freeBlocks
		resp.Bavail = freeBlocks
		resp.HasQuota = true
	}

	return resp, nil
}
