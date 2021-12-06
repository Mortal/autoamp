import argparse
import math
import struct
import wave

import numpy as np
import scipy.signal


WAVE_FORMAT_FL32 = 0x0003


class WaveRead(wave.Wave_read):
    def _read_fmt_chunk(self, chunk):
        try:
            wFormatTag, self._nchannels, self._framerate, dwAvgBytesPerSec, wBlockAlign = struct.unpack_from(
                "<HHLLH", chunk.read(14)
            )
        except struct.error:
            raise EOFError from None
        if wFormatTag == wave.WAVE_FORMAT_PCM:
            try:
                sampwidth = struct.unpack_from("<H", chunk.read(2))[0]
            except struct.error:
                raise EOFError from None
            self._sampwidth = (sampwidth + 7) // 8
            if not self._sampwidth:
                raise wave.Error("bad sample width")
            self._framesize = self._nchannels * self._sampwidth
            self._comptype = "NONE"
            self._compname = "not compressed"
        elif wFormatTag == WAVE_FORMAT_FL32:
            try:
                sampwidth = struct.unpack_from("<H", chunk.read(2))[0]
                extSizeData = chunk.read(2)
                extSize = struct.unpack_from("<H", extSizeData)[0] if extSizeData else 0
            except struct.error:
                raise
                raise EOFError from None
            self._sampwidth = (sampwidth + 7) // 8
            if not self._sampwidth:
                raise wave.Error("bad sample width")
            assert extSize == 0, extSize
            assert chunk.read() == b""
            self._framesize = self._nchannels * self._sampwidth
            self._comptype = "FL32"
            self._compname = "float32 data"
        else:
            raise wave.Error("unknown format: %r" % (wFormatTag,))
        if not self._nchannels:
            raise wave.Error("bad # of channels")


class WaveWrite(wave.Wave_write):
    def setcomptype(self, comptype, compname):
        if self._datawritten:
            raise wave.Error("cannot change parameters after starting to write")
        if comptype not in ("NONE", "FL32"):
            raise wave.Error("unsupported compression type")
        self._comptype = comptype
        self._compname = compname

    def _write_header(self, initlength):
        assert not self._headerwritten
        self._file.write(b"RIFF")
        if not self._nframes:
            self._nframes = initlength // (self._nchannels * self._sampwidth)
        self._datalength = self._nframes * self._nchannels * self._sampwidth
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        self._file.write(
            struct.pack(
                "<L4s4sLHHLLHH4s",
                36 + self._datalength,
                b"WAVE",
                b"fmt ",
                16,
                # wave.py uses WAVE_FORMAT_PCM here
                WAVE_FORMAT_FL32,
                self._nchannels,
                self._framerate,
                self._nchannels * self._framerate * self._sampwidth,
                self._nchannels * self._sampwidth,
                self._sampwidth * 8,
                b"data",
            )
        )
        if self._form_length_pos is not None:
            self._data_length_pos = self._file.tell()
        self._file.write(struct.pack("<L", self._datalength))
        self._headerwritten = True


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--samplerate", type=int)


def rolling(a, window, stride):
    """
    >>> print(rolling(np.r_[0:20], 2, 1).T)
    [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18]
     [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]]
    >>> print(rolling(np.r_[0:20], 2, 2).T)
    [[ 0  2  4  6  8 10 12 14 16 18]
     [ 1  3  5  7  9 11 13 15 17 19]]
    >>> print(rolling(np.r_[0:20], 2, 3).T)
    [[ 0  3  6  9 12 15 18]
     [ 1  4  7 10 13 16 19]]
    >>> print(rolling(np.r_[0:40], 2, 4).T)
    [[ 0  4  8 12 16 20 24 28 32 36]
     [ 1  5  9 13 17 21 25 29 33 37]]
    """
    n, = a.shape
    s, = a.strides
    m = (n - (window - 1) + (stride - 1)) // stride
    a = a[: m * stride + window]
    r = np.lib.stride_tricks.as_strided(a, shape=(m, window), strides=(s * stride, s))
    return r


def rolling_max(a, window):
    return rolling(a, window, 1).max(axis=1)


def repsine(a, mult, freq):
    print(a.shape)
    print(a.shape[0] * mult)
    dbg = np.repeat(a, mult)
    dbg *= np.sin(np.arange(len(dbg)) * (2 * np.pi * freq), dtype=np.float32)
    return dbg


def main() -> None:
    args = parser.parse_args()
    inp = WaveRead(args.input)
    params = inp.getparams()
    assert params.comptype == "FL32"
    data = np.frombuffer(inp.readframes(inp.getnframes()), np.float32).reshape(
        -1, params.nchannels
    )
    print(data.dtype, data.shape)
    samplerate = args.samplerate or params.framerate
    wind = np.hamming(samplerate // 5).astype(np.float32)
    assert not np.any(wind < 0)
    if params.framerate != samplerate:
        g = math.gcd(params.framerate, samplerate)
        data = scipy.signal.resample_poly(
            data, samplerate * g, params.framerate * g
        ).astype(np.float32)
    print(data.dtype, data.shape)
    power = data ** 2
    power = power.max(axis=1)
    k = 4
    r = 3
    w = 20
    pad = samplerate // k * w * k
    power = np.r_[np.zeros(pad // 2, np.float32), power, np.zeros(pad // 2)]
    power_ds = rolling(power, samplerate // k, samplerate // k).max(axis=1)
    print(power_ds.shape, power_ds.dtype)
    power_ds_roll = rolling(power_ds, w * k, r * k)
    print(power_ds_roll.shape, power_ds_roll.dtype)
    lows, medians, highs = np.percentile(power_ds_roll, [20, 70, 95], axis=1).astype(
        np.float32
    )
    print(lows.shape, lows.dtype)
    dbg = np.c_[
        repsine(lows, r * samplerate, 220 / samplerate),
        repsine(medians, r * samplerate, 220 / samplerate),
        repsine(highs, r * samplerate, 220 / samplerate),
    ]
    print(dbg.dtype, dbg.shape)
    out = WaveWrite(args.output)
    out.setparams(params._replace(nchannels=dbg.shape[1], framerate=samplerate))
    out.writeframes(dbg.reshape(-1).tobytes())
    out.close()


if __name__ == "__main__":
    main()
