use {
    anyhow::{anyhow, Result}, log::{debug, trace}, std::fs
};

struct Lump {
    filepos: u32,
    size: u32,
    name: [char;8],
}
impl Lump {
    fn from_raw_bytes(bytes: &[u8; 16]) -> Self {
        let filepos = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let name: [char;8] = [
            bytes[8] as char,
            bytes[9] as char,
            bytes[10] as char,
            bytes[11] as char,
            bytes[12] as char,
            bytes[13] as char,
            bytes[14] as char,
            bytes[15] as char
        ];
        Lump {
            filepos,size,name
        }
    }
}

enum WadType {
    Internal,
    Patch
}
struct Wad {
    wad_type: WadType,
    lumps: Vec<Lump>,
}

impl Wad {
    fn from_file(path: &str) -> Result<Self> {
        let mut content = fs::read(path)?.into_iter();

        // Header check
        match content.by_ref().take(4).collect::<Vec<u8>>().as_slice() {
            [b'P',b'W',b'A',b'D'] => {
                debug!("File header : PWAD");
            },
            [b'I',b'W',b'A',b'D'] => {
                debug!("File header : PWAD");
            },
            _ => return Err(anyhow!("Invalid magic bytes : expected PWAD or IWAD")),
        }

        let numlumps_bytes = content.by_ref().take(4).collect::<Vec<u8>>();
        let numlumps = i32::from_le_bytes([numlumps_bytes[0], numlumps_bytes[1], numlumps_bytes[2], numlumps_bytes[3]]);
        trace!("{path}'s numlumps : {numlumps}");
        let infotableofs_bytes = content.by_ref().take(4).collect::<Vec<u8>>();
        let infotableofs = i32::from_le_bytes([infotableofs_bytes[0], infotableofs_bytes[1], infotableofs_bytes[2], infotableofs_bytes[3]]);
        trace!("{path}'s infotableofs : {infotableofs:X}");


        todo!()
        /*
        Wad {
            lumps: Vec::new(),
        }
        */
    }
}
