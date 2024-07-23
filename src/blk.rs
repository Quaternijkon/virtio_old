use super::*;
use crate::header::VirtIOHeader;
use crate::queue::VirtQueue;
use bitflags::*;
// use core::error::Request;
use core::hint::spin_loop;
use log::*;
use volatile::Volatile;
use zerocopy::{AsBytes, FromBytes, FromZeroes};

const QUEUE: u16 = 0;
const QUEUE_SIZE: u16 = 16;
const SUPPORTED_FEATURES: BlkFeature = BlkFeature::RO //支持只读
    .union(BlkFeature::FLUSH) //支持缓存刷新命令
    .union(BlkFeature::RING_INDIRECT_DESC) //支持间接描述符
    // .union(BlkFeature::RING_EVENT_IDX) //支持事件索引
    .union(BlkFeature::BLK_SIZE) //支持设置块大小
    .union(BlkFeature::CONFIG_WCE) //支持writeback和writethrough模式
    .union(BlkFeature::TOPOLOGY) //支持提供最佳I/O对齐信息
    .union(BlkFeature::DISCARD) //支持丢弃命令
    .union(BlkFeature::WRITE_ZEROES) //支持写零值数据命令
    .union(BlkFeature::MQ) //支持多队列
    .union(BlkFeature::SECURE_ERASE) //支持安全擦除命令
    .union(BlkFeature::ZONED); //支持分区存储设备

/// The virtio block device is a simple virtual block device (ie. disk).
///
/// Read and write requests (and other exotic requests) are placed in the queue,
/// and serviced (probably out of order) by the device except where noted.
pub struct VirtIOBlk<'a, H: Hal> {
    header: &'static mut VirtIOHeader,
    queue: VirtQueue<'a, H>,
    capacity: usize,
    negotiated_features: BlkFeature,
}

impl<H: Hal> VirtIOBlk<'_, H> {
    /// Create a new VirtIO-Blk driver.
    pub fn new(header: &'static mut VirtIOHeader) -> Result<Self> {
        let negotiated_features = BlkFeature::from_bits_truncate(header.begin_init(|features| {
            let features = BlkFeature::from_bits_truncate(features);
            info!("device features: {:?}", features);
            // negotiate these flags only
            let supported_features = SUPPORTED_FEATURES;
            (features & supported_features).bits()
        }));

        // let negotiated_features = header.device_features & SUPPORTED_FEATURES;

        // let negotiated_features = header.begin_init(SUPPORTED_FEATURES);

        // read configuration space
        let config = unsafe { &mut *(header.config_space() as *mut BlkConfig) };
        info!("config: {:?}", config);
        info!(
            "found a block device of size {}KB",
            config.capacity.read() / 2
        );

        let queue = VirtQueue::new(header, 0, 16)?;
        header.finish_init();

        Ok(VirtIOBlk {
            header,
            queue,
            capacity: config.capacity.read() as usize,
            negotiated_features,
        })
    }

    /// 获取块设备的容量, in 512 byte ([`SECTOR_SIZE`]) sectors.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 如果块设备是只读的，则返回 true；如果允许写入，则返回 false。.
    pub fn readonly(&self) -> bool {
        self.negotiated_features.contains(BlkFeature::RO)
    }

    /// Acknowledge interrupt.
    pub fn ack_interrupt(&mut self) -> bool {
        self.header.ack_interrupt()
    }

    // fn request(&mut self, request: BlkReq) -> Result {
    //     let mut resp = BlkResp::default();
    //     self.queue
    // }

    /// Read a block.
    pub fn read_block(&mut self, block_id: usize, buf: &mut [u8]) -> Result {
        assert_eq!(buf.len(), BLK_SIZE);
        let req = BlkReq {
            type_: ReqType::In,
            reserved: 0,
            sector: block_id as u64,
        };
        let mut resp = BlkResp::default();
        self.queue.add(&[req.as_buf()], &[buf, resp.as_buf_mut()])?;
        self.header.notify(0);
        while !self.queue.can_pop() {
            spin_loop();
        }
        self.queue.pop_used()?;
        match resp.status {
            RespStatus::OK => Ok(()),
            _ => Err(Error::IoError),
        }
    }

    /// Read a block in a non-blocking way which means that it returns immediately.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The identifier of the block to read.
    /// * `buf` - The buffer in the memory which the block is read into.
    /// * `resp` - A mutable reference to a variable provided by the caller
    ///   which contains the status of the requests. The caller can safely
    ///   read the variable only after the request is ready.
    ///
    /// # Usage
    ///
    /// It will submit request to the virtio block device and return a token identifying
    /// the position of the first Descriptor in the chain. If there are not enough
    /// Descriptors to allocate, then it returns [Error::BufferTooSmall].
    ///
    /// After the request is ready, `resp` will be updated and the caller can get the
    /// status of the request(e.g. succeed or failed) through it. However, the caller
    /// **must not** spin on `resp` to wait for it to change. A safe way is to read it
    /// after the same token as this method returns is fetched through [VirtIOBlk::pop_used()],
    /// which means that the request has been ready.
    ///
    /// # Safety
    ///
    /// `buf` is still borrowed by the underlying virtio block device even if this
    /// method returns. Thus, it is the caller's responsibility to guarantee that
    /// `buf` is not accessed before the request is completed in order to avoid
    /// data races.
    pub unsafe fn read_block_nb(
        &mut self,
        block_id: usize,
        buf: &mut [u8],
        resp: &mut BlkResp,
    ) -> Result<u16> {
        assert_eq!(buf.len(), BLK_SIZE);
        let req = BlkReq {
            type_: ReqType::In,
            reserved: 0,
            sector: block_id as u64,
        };
        let token = self.queue.add(&[req.as_buf()], &[buf, resp.as_buf_mut()])?;
        self.header.notify(0);
        Ok(token)
    }

    /// Write a block.
    pub fn write_block(&mut self, block_id: usize, buf: &[u8]) -> Result {
        assert_eq!(buf.len(), BLK_SIZE);
        let req = BlkReq {
            type_: ReqType::Out,
            reserved: 0,
            sector: block_id as u64,
        };
        let mut resp = BlkResp::default();
        self.queue.add(&[req.as_buf(), buf], &[resp.as_buf_mut()])?;
        self.header.notify(0);
        while !self.queue.can_pop() {
            spin_loop();
        }
        self.queue.pop_used()?;
        match resp.status {
            RespStatus::OK => Ok(()),
            _ => Err(Error::IoError),
        }
    }

    //// Write a block in a non-blocking way which means that it returns immediately.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The identifier of the block to write.
    /// * `buf` - The buffer in the memory containing the data to write to the block.
    /// * `resp` - A mutable reference to a variable provided by the caller
    ///   which contains the status of the requests. The caller can safely
    ///   read the variable only after the request is ready.
    ///
    /// # Usage
    ///
    /// See also [VirtIOBlk::read_block_nb()].
    ///
    /// # Safety
    ///
    /// See also [VirtIOBlk::read_block_nb()].
    pub unsafe fn write_block_nb(
        &mut self,
        block_id: usize,
        buf: &[u8],
        resp: &mut BlkResp,
    ) -> Result<u16> {
        assert_eq!(buf.len(), BLK_SIZE);
        let req = BlkReq {
            type_: ReqType::Out,
            reserved: 0,
            sector: block_id as u64,
        };
        let token = self.queue.add(&[req.as_buf(), buf], &[resp.as_buf_mut()])?;
        self.header.notify(0);
        Ok(token)
    }

    /// During an interrupt, it fetches a token of a completed request from the used
    /// ring and return it. If all completed requests have already been fetched, return
    /// Err(Error::NotReady).
    pub fn pop_used(&mut self) -> Result<u16> {
        self.queue.pop_used().map(|p| p.0)
    }

    /// Return size of its VirtQueue.
    /// It can be used to tell the caller how many channels he should monitor on.
    pub fn virt_queue_size(&self) -> u16 {
        self.queue.size()
    }
}

#[repr(C)]
#[derive(Debug)]
struct BlkConfig {
    /// Number of 512 Bytes sectors
    capacity: Volatile<u64>,
    size_max: Volatile<u32>,
    seg_max: Volatile<u32>,
    cylinders: Volatile<u16>,
    heads: Volatile<u8>,
    sectors: Volatile<u8>,
    blk_size: Volatile<u32>,
    physical_block_exp: Volatile<u8>,
    alignment_offset: Volatile<u8>,
    min_io_size: Volatile<u16>,
    opt_io_size: Volatile<u32>,
    writeback: Volatile<u8>,
    unused0: Volatile<u8>,
    num_queues: Volatile<u16>,
    max_discard_sectors: Volatile<u32>,
    max_discard_seg: Volatile<u32>,
    discard_sector_alignment: Volatile<u32>,
    max_write_zeroes_sectors: Volatile<u32>,
    max_write_zeroes_seg: Volatile<u32>,
    write_zeroes_may_unmap: Volatile<u32>,
    unused1: [Volatile<u8>; 3],
    max_secure_erase_sectors: Volatile<u32>,
    max_secure_erase_seg: Volatile<u32>,
    secure_erase_sector_alignment: Volatile<u32>,
    zone_sectors: Volatile<u32>,
    max_open_zones: Volatile<u32>,
    max_active_zones: Volatile<u32>,
    max_append_sectors: Volatile<u32>,
    write_granularity: Volatile<u32>,
    model: ModelType, //enum ModelType
    unused2: [Volatile<u8>; 3],
}

#[repr(u8)]
#[derive(AsBytes, Debug)]
enum ModelType {
    NONE = 0,
    HM = 1,
    HA = 2,
}

#[repr(C)]
#[derive(Debug)]
struct BlkReq {
    type_: ReqType,
    reserved: u32,
    sector: u64,
}

impl Default for BlkReq {
    fn default() -> Self {
        Self {
            type_: ReqType::In,
            reserved: 0,
            sector: 0,
            // data: [0; 16],
            // status: StatusType::OK,
        }
    }
}

/// Response of a VirtIOBlk request.
#[repr(C)]
#[derive(AsBytes, Debug, FromBytes, FromZeroes)]
pub struct BlkResp {
    status: RespStatus,
}

impl BlkResp {
    /// Return the status of a VirtIOBlk request.
    pub fn status(&self) -> RespStatus {
        self.status
    }
}

#[repr(u32)]
#[derive(Debug)]
enum ReqType {
    In = 0,
    Out = 1,
    Flush = 4,
    GetId = 8,
    GetLifetime = 10,
    Discard = 11,
    WriteZeroes = 13,
    SecureErase = 14,

    Append = 15,
    Report = 16,
    Open = 18,
    Close = 20,
    Finish = 22,
    Reset = 24,
    ResetAll = 26,
}

/// Status of a VirtIOBlk request.
// #[repr(u8)]
// #[derive(Debug, Eq, PartialEq, Copy, Clone)]
// pub enum RespStatus {
//     /// Ok.
//     Ok = 0,
//     /// IoErr.
//     IoErr = 1,
//     /// Unsupported yet.
//     Unsupported = 2,
//     /// Not ready.
//     _NotReady = 3,
// }

struct BlkReqZoneAppend {
    type_: ReqTypeZoneAppend,
    reserved: u32,
    sector: u64,
    data: [u8; 16],
    append_sectors: u64,
    status: u8,
}

#[repr(u32)]
#[derive(AsBytes, Debug)]
enum ReqTypeZoneAppend {
    InvalidCMD = 3,
    UnalignedWP = 4,
    OpenResource = 5,
    ActiveResource = 6,
}

struct BlkZoneReport {
    nr_zones: u64,
    reserved: [u8; 56],
    zones: [BlkZoneDescriptor; 32],
}

struct BlkZoneDescriptor {
    z_cap: u64,
    z_start: u64,
    z_wp: u64,
    z_type: ZoneType,
    z_state: ZoneStateType,
    reserved: [u8; 38],
}

#[repr(u8)]
#[derive(AsBytes, Debug)]
enum ZoneType {
    CONV = 1,
    SWR = 2,
    SWP = 3,
}

#[repr(u32)]
#[derive(AsBytes, Debug)]
enum ZoneStateType {
    NotWP = 0,
    Empty = 1,
    IOPEN = 2,
    EOPEN = 3,
    Closed = 4,
    RdOnly = 13,
    Full = 14,
    Offline = 15,
}

#[repr(transparent)]
#[derive(AsBytes, Copy, Clone, Debug, Eq, FromBytes, FromZeroes, PartialEq)]
pub struct RespStatus(u8);

impl RespStatus {
    /// Ok.
    pub const OK: RespStatus = RespStatus(0);
    /// IoErr.
    pub const IO_ERR: RespStatus = RespStatus(1);
    /// Unsupported yet.
    pub const UNSUPPORTED: RespStatus = RespStatus(2);
    /// Not ready.
    pub const NOT_READY: RespStatus = RespStatus(3);
}

impl From<RespStatus> for Result {
    fn from(status: RespStatus) -> Self {
        match status {
            RespStatus::OK => Ok(()),
            RespStatus::IO_ERR => Err(Error::IoError),
            RespStatus::UNSUPPORTED => Err(Error::Unsupported),
            RespStatus::NOT_READY => Err(Error::NotReady),
            _ => Err(Error::IoError),
        }
    }
}

impl Default for BlkResp {
    fn default() -> Self {
        BlkResp {
            status: RespStatus::NOT_READY,
        }
    }
}

struct BlkDiscardWriteZeroes {
    sector: u64,
    num_sectors: u32,
    flags: Flags, //struct Flags
}

struct Flags {
    unmap: u32,
    reserved: u32,
}

impl Default for Flags {
    fn default() -> Self {
        Flags {
            unmap: 1,
            reserved: 31,
        }
    }
}

struct BlkLifetime {
    pre_eol_info: PreEolInfoType,
    est_typ_a: u16,
    est_typ_b: u16,
}

#[repr(u8)]
#[derive(AsBytes, Debug)]
enum PreEolInfoType {
    Undefined = 0,
    Normal = 1,
    Warning = 2,
    Urgent = 3,
}

#[repr(u64)]
#[derive(AsBytes, Debug)]
enum StatusType {
    OK = 0,
    IOErr = 1,
    UnSupp = 2,
}

const SCSI_SENSE_BUFFERSIZE: usize = 96;

/// All fields are in guest's native endian.

struct ScsiPcReq {
    type_: ScsiPcReqType,
    ioprio: u32,
    sector: u64,
    cmd: [u8; 16],
    data: [u8; 512],
    sense: [u8; SCSI_SENSE_BUFFERSIZE],
    error: u32,
    data_len: u32,
    sense_len: u32,
    residual: u32,
    status: u8,
}

#[repr(u64)]
#[derive(AsBytes, Debug)]
enum ScsiPcReqType {
    SCSICMD = 2,
    SCSICMDOUT = 3,
}

const BLK_SIZE: usize = 512;

bitflags! {
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    struct BlkFeature: u64 {
        /// 设备支持请求屏障 (legacy)
        const BARRIER       = 1 << 0;
        /// 任何单个段的最大大小在 `size_max` 中。
        const SIZE_MAX      = 1 << 1;
        /// 一个请求中的最大段数在 `seg_max` 中。
        const SEG_MAX       = 1 << 2;
        /// 在 geometry 中指定了磁盘样式的几何结构。
        const GEOMETRY      = 1 << 4;
        /// 设备是只读的。
        const RO            = 1 << 5;
        /// 磁盘的块大小在 `blk_size` 中。
        const BLK_SIZE      = 1 << 6;
        /// 设备支持 SCSI 包命令（legacy）。
        const SCSI          = 1 << 7;
        /// 缓存刷新命令支持。
        const FLUSH         = 1 << 9;
        /// 设备提供有关最佳 I/O 对齐的信息。
        const TOPOLOGY      = 1 << 10;
        /// 设备可以在写回和写穿透模式之间切换其缓存。
        const CONFIG_WCE    = 1 << 11;
        /// 设备支持多队列。
        const MQ            = 1 << 12;
        /// 设备可以支持丢弃命令，在 `max_discard_sectors` 中指定最大丢弃扇区大小，
        /// 在 `max_discard_seg` 中指定最大丢弃段数。
        const DISCARD       = 1 << 13;
        /// 设备可以支持写零命令，在 `max_write_zeroes_sectors` 中指定最大写零扇区大小，
        /// 在 `max_write_zeroes_seg` 中指定最大写零段数。
        const WRITE_ZEROES  = 1 << 14;
        /// 设备支持提供存储寿命信息。
        const LIFETIME      = 1 << 15;
        /// 设备可以支持安全擦除命令。
        const SECURE_ERASE  = 1 << 16;
        /// 设备是遵循分区存储的设备。
        const ZONED         = 1 << 17;

        // 设备独立的特性
        const NOTIFY_ON_EMPTY       = 1 << 24; // legacy
        const ANY_LAYOUT            = 1 << 27; // legacy
        const RING_INDIRECT_DESC    = 1 << 28;
        const RING_EVENT_IDX        = 1 << 29;
        const UNUSED                = 1 << 30; // legacy
        const VERSION_1             = 1 << 32; // detect legacy

        // 自 VirtIO v1.1 起支持以下功能。
        const ACCESS_PLATFORM       = 1 << 33;
        const RING_PACKED           = 1 << 34;
        const IN_ORDER              = 1 << 35;
        const ORDER_PLATFORM        = 1 << 36;
        const SR_IOV                = 1 << 37;
        const NOTIFICATION_DATA     = 1 << 38;
    }
}

unsafe impl AsBuf for BlkReq {}
unsafe impl AsBuf for BlkResp {}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::{
//         hal::fake::FakeHal,
//         transport::{
//             fake::{FakeTransport, QueueStatus, State},
//             DeviceType,
//         },
//     };
//     use alloc::{sync::Arc, vec};
//     use core::{mem::size_of, ptr::NonNull};
//     use std::{sync::Mutex, thread};

//     #[test]
//     fn config() {
//         let mut config_space = BlkConfig {
//             capacity_low: Volatile::new(0x42),
//             capacity_high: Volatile::new(0x02),
//             size_max: Volatile::new(0),
//             seg_max: Volatile::new(0),
//             cylinders: Volatile::new(0),
//             heads: Volatile::new(0),
//             sectors: Volatile::new(0),
//             blk_size: Volatile::new(0),
//             physical_block_size: Volatile::new(0),
//             alignment_offset: Volatile::new(0),
//             min_io_size: Volatile::new(0),
//             opt_io_size: Volatile::new(0),
//             writeback: Volatile::new(0),
//             unused0: Volatile::new(0),
//             num_queues: Volatile::new(0),
//             max_discard_sectors: Volatile::new(0),
//             max_discard_seg: Volatile::new(0),
//             discard_sector_alignment: Volatile::new(0),
//             max_write_zeroes_sectors: Volatile::new(0),
//             max_write_zeroes_seg: Volatile::new(0),
//             write_zeroes_may_unmap: Volatile::new(0),
//             unused1: [Volatile::new(0); 3],
//             max_secure_erase_sectors: Volatile::new(0),
//             max_secure_erase_seg: Volatile::new(0),
//             secure_erase_sector_alignment: Volatile::new(0),
//             zone_sectors: Volatile::new(0),
//             max_open_zones: Volatile::new(0),
//             max_active_zones: Volatile::new(0),
//             max_append_sectors: Volatile::new(0),
//             write_granularity: Volatile::new(0),
//             model: ModelType::NONE,
//             unused2: [Volatile::new(0); 3],
//         };
//         let state = Arc::new(Mutex::new(State {
//             queues: vec![QueueStatus::default()],
//             ..Default::default()
//         }));
//         let transport = FakeTransport {
//             device_type: DeviceType::Block,
//             max_queue_size: QUEUE_SIZE.into(),
//             device_features: BlkFeature::RO.bits(),
//             config_space: NonNull::from(&mut config_space),
//             state: state.clone(),
//         };
//         let blk = VirtIOBlk::<FakeHal, FakeTransport<BlkConfig>>::new(transport).unwrap();

//         assert_eq!(blk.capacity(), 0x02_0000_0042);
//         assert_eq!(blk.readonly(), true);
//     }

//     #[test]
//     fn read() {
//         let mut config_space = BlkConfig {
//             capacity_low: Volatile::new(66),
//             capacity_high: Volatile::new(0),
//             size_max: Volatile::new(0),
//             seg_max: Volatile::new(0),
//             geometry: BlkGeometry {
//                 cylinders: Volatile::new(0),
//                 heads: Volatile::new(0),
//                 sectors: Volatile::new(0),
//             },
//             blk_size: Volatile::new(0),
//             topology: BlkTopology {
//                 physical_block_size: Volatile::new(0),
//                 alignment_offset: Volatile::new(0),
//                 min_io_size: Volatile::new(0),
//                 opt_io_size: Volatile::new(0),
//             },
//             writeback: Volatile::new(0),
//             unused0: Volatile::new(0),
//             num_queues: Volatile::new(0),
//             max_discard_sectors: Volatile::new(0),
//             max_discard_seg: Volatile::new(0),
//             discard_sector_alignment: Volatile::new(0),
//             max_write_zeroes_sectors: Volatile::new(0),
//             max_write_zeroes_seg: Volatile::new(0),
//             write_zeroes_may_unmap: Volatile::new(0),
//             unused1: [Volatile::new(0); 3],
//             max_secure_erase_sectors: Volatile::new(0),
//             max_secure_erase_seg: Volatile::new(0),
//             secure_erase_sector_alignment: Volatile::new(0),
//             zoned: BlkZonedCharacteristics {
//                 zone_sectors: Volatile::new(0),
//                 max_open_zones: Volatile::new(0),
//                 max_active_zones: Volatile::new(0),
//                 max_append_sectors: Volatile::new(0),
//                 write_granularity: Volatile::new(0),
//                 model: ModelType::NONE,
//                 unused2: [Volatile::new(0); 3],
//             },
//         };
//         let state = Arc::new(Mutex::new(State {
//             queues: vec![QueueStatus::default()],
//             ..Default::default()
//         }));
//         let transport = FakeTransport {
//             device_type: DeviceType::Block,
//             max_queue_size: QUEUE_SIZE.into(),
//             device_features: BlkFeature::RING_INDIRECT_DESC.bits(),
//             config_space: NonNull::from(&mut config_space),
//             state: state.clone(),
//         };
//         let mut blk = VirtIOBlk::<FakeHal, FakeTransport<BlkConfig>>::new(transport).unwrap();

//         // Start a thread to simulate the device waiting for a read request.
//         let handle = thread::spawn(move || {
//             println!("Device waiting for a request.");
//             State::wait_until_queue_notified(&state, QUEUE);
//             println!("Transmit queue was notified.");

//             state
//                 .lock()
//                 .unwrap()
//                 .read_write_queue::<{ QUEUE_SIZE as usize }>(QUEUE, |request| {
//                     assert_eq!(
//                         request,
//                         BlkReq {
//                             type_: ReqType::In,
//                             reserved: 0,
//                             sector: 42,
//                             // data: [0; 16],
//                             // status: StatusType::OK,
//                         }
//                         .as_bytes()
//                     );

//                     let mut response = vec![0; SECTOR_SIZE];
//                     response[0..9].copy_from_slice(b"Test data");
//                     response.extend_from_slice(
//                         BlkResp {
//                             status: RespStatus::OK,
//                         }
//                         .as_bytes(),
//                     );

//                     response
//                 });
//         });

//         // Read a block from the device.
//         let mut buffer = [0; 512];
//         blk.read_blocks(42, &mut buffer).unwrap();
//         assert_eq!(&buffer[0..9], b"Test data");

//         handle.join().unwrap();
//     }

//     #[test]
//     fn write() {
//         let mut config_space = BlkConfig {
//             capacity_low: Volatile::new(66),
//             capacity_high: Volatile::new(0),
//             size_max: Volatile::new(0),
//             seg_max: Volatile::new(0),
//             geometry: BlkGeometry {
//                 cylinders: Volatile::new(0),
//                 heads: Volatile::new(0),
//                 sectors: Volatile::new(0),
//             },
//             blk_size: Volatile::new(0),
//             topology: BlkTopology {
//                 physical_block_size: Volatile::new(0),
//                 alignment_offset: Volatile::new(0),
//                 min_io_size: Volatile::new(0),
//                 opt_io_size: Volatile::new(0),
//             },
//             writeback: Volatile::new(0),
//             unused0: Volatile::new(0),
//             num_queues: Volatile::new(0),
//             max_discard_sectors: Volatile::new(0),
//             max_discard_seg: Volatile::new(0),
//             discard_sector_alignment: Volatile::new(0),
//             max_write_zeroes_sectors: Volatile::new(0),
//             max_write_zeroes_seg: Volatile::new(0),
//             write_zeroes_may_unmap: Volatile::new(0),
//             unused1: [Volatile::new(0); 3],
//             max_secure_erase_sectors: Volatile::new(0),
//             max_secure_erase_seg: Volatile::new(0),
//             secure_erase_sector_alignment: Volatile::new(0),
//             zoned: BlkZonedCharacteristics {
//                 zone_sectors: Volatile::new(0),
//                 max_open_zones: Volatile::new(0),
//                 max_active_zones: Volatile::new(0),
//                 max_append_sectors: Volatile::new(0),
//                 write_granularity: Volatile::new(0),
//                 model: ModelType::NONE,
//                 unused2: [Volatile::new(0); 3],
//             },
//         };
//         let state = Arc::new(Mutex::new(State {
//             queues: vec![QueueStatus::default()],
//             ..Default::default()
//         }));
//         let transport = FakeTransport {
//             device_type: DeviceType::Block,
//             max_queue_size: QUEUE_SIZE.into(),
//             device_features: BlkFeature::RING_INDIRECT_DESC.bits(),
//             config_space: NonNull::from(&mut config_space),
//             state: state.clone(),
//         };
//         let mut blk = VirtIOBlk::<FakeHal, FakeTransport<BlkConfig>>::new(transport).unwrap();

//         // Start a thread to simulate the device waiting for a write request.
//         let handle = thread::spawn(move || {
//             println!("Device waiting for a request.");
//             State::wait_until_queue_notified(&state, QUEUE);
//             println!("Transmit queue was notified.");

//             state
//                 .lock()
//                 .unwrap()
//                 .read_write_queue::<{ QUEUE_SIZE as usize }>(QUEUE, |request| {
//                     assert_eq!(
//                         &request[0..size_of::<BlkReq>()],
//                         BlkReq {
//                             type_: ReqType::Out,
//                             reserved: 0,
//                             sector: 42,
//                             // data: [0; 16],
//                             // status: StatusType::OK,
//                         }
//                         .as_bytes()
//                     );
//                     let data = &request[size_of::<BlkReq>()..];
//                     assert_eq!(data.len(), SECTOR_SIZE);
//                     assert_eq!(&data[0..9], b"Test data");

//                     let mut response = Vec::new();
//                     response.extend_from_slice(
//                         BlkResp {
//                             status: RespStatus::OK,
//                         }
//                         .as_bytes(),
//                     );

//                     response
//                 });
//         });

//         // Write a block to the device.
//         let mut buffer = [0; 512];
//         buffer[0..9].copy_from_slice(b"Test data");
//         blk.write_blocks(42, &mut buffer).unwrap();

//         // Request to flush should be ignored as the device doesn't support it.
//         blk.flush().unwrap();

//         handle.join().unwrap();
//     }

//     #[test]
//     fn flush() {
//         let mut config_space = BlkConfig {
//             capacity_low: Volatile::new(66),
//             capacity_high: Volatile::new(0),
//             size_max: Volatile::new(0),
//             seg_max: Volatile::new(0),
//             geometry: BlkGeometry {
//                 cylinders: Volatile::new(0),
//                 heads: Volatile::new(0),
//                 sectors: Volatile::new(0),
//             },
//             blk_size: Volatile::new(0),
//             topology: BlkTopology {
//                 physical_block_size: Volatile::new(0),
//                 alignment_offset: Volatile::new(0),
//                 min_io_size: Volatile::new(0),
//                 opt_io_size: Volatile::new(0),
//             },
//             writeback: Volatile::new(0),
//             unused0: Volatile::new(0),
//             num_queues: Volatile::new(0),
//             max_discard_sectors: Volatile::new(0),
//             max_discard_seg: Volatile::new(0),
//             discard_sector_alignment: Volatile::new(0),
//             max_write_zeroes_sectors: Volatile::new(0),
//             max_write_zeroes_seg: Volatile::new(0),
//             write_zeroes_may_unmap: Volatile::new(0),
//             unused1: [Volatile::new(0); 3],
//             max_secure_erase_sectors: Volatile::new(0),
//             max_secure_erase_seg: Volatile::new(0),
//             secure_erase_sector_alignment: Volatile::new(0),
//             zoned: BlkZonedCharacteristics {
//                 zone_sectors: Volatile::new(0),
//                 max_open_zones: Volatile::new(0),
//                 max_active_zones: Volatile::new(0),
//                 max_append_sectors: Volatile::new(0),
//                 write_granularity: Volatile::new(0),
//                 model: ModelType::NONE,
//                 unused2: [Volatile::new(0); 3],
//             },
//         };
//         let state = Arc::new(Mutex::new(State {
//             queues: vec![QueueStatus::default()],
//             ..Default::default()
//         }));
//         let transport = FakeTransport {
//             device_type: DeviceType::Block,
//             max_queue_size: QUEUE_SIZE.into(),
//             device_features: (BlkFeature::RING_INDIRECT_DESC | BlkFeature::FLUSH).bits(),
//             config_space: NonNull::from(&mut config_space),
//             state: state.clone(),
//         };
//         let mut blk = VirtIOBlk::<FakeHal, FakeTransport<BlkConfig>>::new(transport).unwrap();

//         // Start a thread to simulate the device waiting for a flush request.
//         let handle = thread::spawn(move || {
//             println!("Device waiting for a request.");
//             State::wait_until_queue_notified(&state, QUEUE);
//             println!("Transmit queue was notified.");

//             state
//                 .lock()
//                 .unwrap()
//                 .read_write_queue::<{ QUEUE_SIZE as usize }>(QUEUE, |request| {
//                     assert_eq!(
//                         request,
//                         BlkReq {
//                             type_: ReqType::Flush,
//                             reserved: 0,
//                             sector: 0,
//                             // data: [0; 16],
//                             // status: StatusType::OK,
//                         }
//                         .as_bytes()
//                     );

//                     let mut response = Vec::new();
//                     response.extend_from_slice(
//                         BlkResp {
//                             status: RespStatus::OK,
//                         }
//                         .as_bytes(),
//                     );

//                     response
//                 });
//         });

//         // Request to flush.
//         blk.flush().unwrap();

//         handle.join().unwrap();
//     }

//     #[test]
//     fn device_id() {
//         let mut config_space = BlkConfig {
//             capacity_low: Volatile::new(66),
//             capacity_high: Volatile::new(0),
//             size_max: Volatile::new(0),
//             seg_max: Volatile::new(0),
//             geometry: BlkGeometry {
//                 cylinders: Volatile::new(0),
//                 heads: Volatile::new(0),
//                 sectors: Volatile::new(0),
//             },
//             blk_size: Volatile::new(0),
//             topology: BlkTopology {
//                 physical_block_size: Volatile::new(0),
//                 alignment_offset: Volatile::new(0),
//                 min_io_size: Volatile::new(0),
//                 opt_io_size: Volatile::new(0),
//             },
//             writeback: Volatile::new(0),
//             unused0: Volatile::new(0),
//             num_queues: Volatile::new(0),
//             max_discard_sectors: Volatile::new(0),
//             max_discard_seg: Volatile::new(0),
//             discard_sector_alignment: Volatile::new(0),
//             max_write_zeroes_sectors: Volatile::new(0),
//             max_write_zeroes_seg: Volatile::new(0),
//             write_zeroes_may_unmap: Volatile::new(0),
//             unused1: [Volatile::new(0); 3],
//             max_secure_erase_sectors: Volatile::new(0),
//             max_secure_erase_seg: Volatile::new(0),
//             secure_erase_sector_alignment: Volatile::new(0),
//             zoned: BlkZonedCharacteristics {
//                 zone_sectors: Volatile::new(0),
//                 max_open_zones: Volatile::new(0),
//                 max_active_zones: Volatile::new(0),
//                 max_append_sectors: Volatile::new(0),
//                 write_granularity: Volatile::new(0),
//                 model: ModelType::NONE,
//                 unused2: [Volatile::new(0); 3],
//             },
//         };
//         let state = Arc::new(Mutex::new(State {
//             queues: vec![QueueStatus::default()],
//             ..Default::default()
//         }));
//         let transport = FakeTransport {
//             device_type: DeviceType::Block,
//             max_queue_size: QUEUE_SIZE.into(),
//             device_features: BlkFeature::RING_INDIRECT_DESC.bits(),
//             config_space: NonNull::from(&mut config_space),
//             state: state.clone(),
//         };
//         let mut blk = VirtIOBlk::<FakeHal, FakeTransport<BlkConfig>>::new(transport).unwrap();

//         // Start a thread to simulate the device waiting for a flush request.
//         let handle = thread::spawn(move || {
//             println!("Device waiting for a request.");
//             State::wait_until_queue_notified(&state, QUEUE);
//             println!("Transmit queue was notified.");

//             state
//                 .lock()
//                 .unwrap()
//                 .read_write_queue::<{ QUEUE_SIZE as usize }>(QUEUE, |request| {
//                     assert_eq!(
//                         request,
//                         BlkReq {
//                             type_: ReqType::GetId,
//                             reserved: 0,
//                             sector: 0,
//                             // data: [0; 16],
//                             // status: StatusType::OK,
//                         }
//                         .as_bytes()
//                     );

//                     let mut response = Vec::new();
//                     response.extend_from_slice(b"device_id\0\0\0\0\0\0\0\0\0\0\0");
//                     response.extend_from_slice(
//                         BlkResp {
//                             status: RespStatus::OK,
//                         }
//                         .as_bytes(),
//                     );

//                     response
//                 });
//         });

//         let mut id = [0; 20];
//         let length = blk.device_id(&mut id).unwrap();
//         assert_eq!(&id[0..length], b"device_id");

//         handle.join().unwrap();
//     }
// }
