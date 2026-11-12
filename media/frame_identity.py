"""Checksummed frame identity in a cropped 64-row video band.

Readers verify this code instead of assuming RTSP/WebRTC clock origins.
The band is removed before inference/presentation. CRC is an identity/error
check, not a cryptographic authenticator.
"""
from __future__ import annotations
import binascii
import json
import struct
import zlib
from dataclasses import dataclass

BAND_HEIGHT = 64
CODE_WIDTH = 128
CELL = 8
MAGIC = b"EV"


def stream_tag(key) -> int:
    return zlib.crc32(json.dumps(list(key), ensure_ascii=True, separators=(",", ":")).encode())


def frame_pts_ns(seq: int, fps: int) -> int:
    return (seq * 1_000_000_000 + fps // 2) // fps


@dataclass(frozen=True)
class EncodedIdentity:
    seq: int
    fps: int
    padding: bool = False

    @property
    def pts_ns(self) -> int:
        return frame_pts_ns(self.seq, self.fps)


def write_identity_band(band, *, key, seq: int, fps: int, padding: bool = False) -> None:
    """Fill a (BAND_HEIGHT, width, 3) uint8 view in place with the identity code."""
    import numpy as np
    if band.shape[1] < CODE_WIDTH or not 0 <= seq < 2**32 or not 0 < fps <= 240 or int(fps) != fps:
        raise ValueError("identity video requires width >=128, integer FPS 1..240 and uint32 sequence")
    body = struct.pack(">2sBBIIH", MAGIC, 1, int(padding), stream_tag(key), seq, int(fps))
    packet = body + struct.pack(">H", binascii.crc_hqx(body, 0xffff))
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8)).reshape(8, 16)
    cells = np.repeat(np.repeat(bits, CELL, axis=0), CELL, axis=1) * 235 + 10
    band[:, :CODE_WIDTH] = cells[:, :, None]
    band[:, CODE_WIDTH:] = 0


def append_identity(pixels, *, key, seq: int, fps: int, padding: bool = False):
    import numpy as np
    band = np.empty((BAND_HEIGHT, pixels.shape[1], 3), dtype=np.uint8)
    write_identity_band(band, key=key, seq=seq, fps=fps, padding=padding)
    return np.concatenate((pixels, band), axis=0)


def read_identity(pixels, *, key) -> EncodedIdentity:
    import numpy as np
    if pixels.shape[0] <= BAND_HEIGHT or pixels.shape[1] < CODE_WIDTH:
        raise ValueError("video has no identity band")
    band = pixels[-BAND_HEIGHT:, :CODE_WIDTH, :3]
    cells = band.reshape(8, CELL, 16, CELL, 3)[:, 2:6, :, 2:6].mean(axis=(1, 3, 4))
    if np.any((cells > 80) & (cells < 175)):
        raise ValueError("ambiguous frame identity cells")
    packet = np.packbits((cells > 127).astype(np.uint8).reshape(-1)).tobytes()
    body = packet[:14]
    if binascii.crc_hqx(body, 0xffff) != int.from_bytes(packet[14:], "big"):
        raise ValueError("frame identity CRC mismatch")
    magic, version, flags, tag, seq, fps = struct.unpack(">2sBBIIH", body)
    if magic != MAGIC or version != 1 or flags not in (0, 1) or tag != stream_tag(key) or not 0 < fps <= 240:
        raise ValueError("frame identity stream/version mismatch")
    return EncodedIdentity(seq, fps, bool(flags))
