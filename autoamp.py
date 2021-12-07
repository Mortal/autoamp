import argparse
import math
import struct
import wave
from typing import List

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


def sections(bits):
    bits = np.r_[False, bits, False]
    starts = bits[1:] & ~bits[:-1]
    ends = ~bits[1:] & bits[:-1]
    assert np.sum(starts) == np.sum(ends)
    return zip(starts.nonzero()[0], ends.nonzero()[0])


def amp_and_duck(samples, target):
    assert samples.dtype == np.float32
    amp = np.ones(len(samples) + 1, dtype=np.float32) / target
    above = samples > target
    amp[:-1][above] = np.minimum(amp[:-1][above], 1 / samples[above])
    amp[1:][above] = np.minimum(amp[1:][above], 1 / samples[above])
    assert len(amp) == len(samples) + 1
    return amp


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
    power = data.max(axis=1)
    bucketsize = samplerate // 4
    power_ds = rolling(power, bucketsize, bucketsize).max(axis=1)
    vals = np.argsort(power_ds)
    n = len(power_ds)
    node = np.zeros((n,), np.int32) - 1
    tree: List[List[int]] = []
    root: List[int] = []
    unions = []

    def find_root(n):
        assert n >= 0
        while root[n] != n:
            root[n] = n = root[root[n]]
        return n

    for i in vals:
        assert node[i] == -1
        left = i > 0 and node[i-1] != -1
        right = i + 1 < n and node[i+1] != -1
        if left and right:
            node[i] = len(tree)
            root.append(len(tree))
            left_root = find_root(node[i-1])
            right_root = find_root(node[i+1])
            root[left_root] = root[right_root] = len(tree)
            left_node = tree[left_root]
            right_node = tree[right_root]
            unions.append((power_ds[i], left_node[:], right_node[:]))
            tree.append([left_node[0], len(tree), right_node[2], min((left_node[3], right_node[3]), key=lambda i: power_ds[i])])
        elif right:
            node[i] = find_root(node[i+1])
            assert tree[node[i]][0] == i + 1
            tree[node[i]][0] = i
        elif left:
            node[i] = find_root(node[i-1])
            assert tree[node[i]][2] == i - 1, (i, node[i], node[i-1], tree[node[i-1]], tree[node[i]])
            tree[node[i]][2] = i
        else:
            node[i] = len(tree)
            root.append(len(tree))
            tree.append([i, i, i, i])

    time_threshold = samplerate * 10 / bucketsize

    for p, a, b in unions:
        a_a = np.inf if a[0] == 0 else a[2] - a[0] + 1
        b_a = np.inf if b[2] == len(power_ds)-1 else b[2] - b[0] + 1
        if b_a < time_threshold:
            assert np.all(power_ds[b[0]:b[2]+1] <= p)
            power_ds[b[0]:b[2]+1] = p
        if a_a < time_threshold:
            assert np.all(power_ds[a[0]:a[2]+1] <= p)
            power_ds[a[0]:a[2]+1] = p
        # if a_a > b_a < time_threshold:
        #     assert np.all(power_ds[b[0]:b[2]+1] < p)
        #     power_ds[b[0]:b[2]+1] = p
        # elif b_a > a_a < time_threshold:
        #     assert np.all(power_ds[a[0]:a[2]+1] < p)
        #     power_ds[a[0]:a[2]+1] = p

    if 0:
        diff = power_ds[1:] != power_ds[:-1]
        i = 0
        f = bucketsize / samplerate
        for j in diff.nonzero()[0]:
            print(f*i, f*(j+1), power_ds[i])
            i = j+1
        print(f*i, f*len(power_ds), power_ds[i])

    if 0:
        for loud_threshold in np.arange(30):
            loud_threshold = (1 + loud_threshold) * 0.001
            is_quiet = power_ds < loud_threshold
            quiet_target = np.percentile(power_ds[is_quiet], 99)
            print("%.3f" % loud_threshold, quiet_target)

    loud_threshold = 0.011 ** 0.5
    is_quiet = power_ds < loud_threshold
    quiet_target = np.median(power_ds[is_quiet])
    print("%.4f" % loud_threshold, quiet_target)
    for i, j in sections(is_quiet & (power_ds > quiet_target)):
        if i == 0 or not is_quiet[i-1] or j == len(is_quiet) or not is_quiet[j]:
            is_quiet[i:j] = False
    # dbg = np.sin(np.arange(len(power)) * (2 * np.pi * 220 / samplerate), dtype=np.float32)
    # for i, j in sections(~is_quiet):
    #     dbg[i * bucketsize : j * bucketsize] = 0
    output = np.array(data)
    print(output.dtype, output.shape)
    for i, j in sections(is_quiet):
        assert len(power_ds[i:j]) == j - i
        assert len(np.arange(j - i + 1)) == j - i + 1
        amt = np.interp(
            np.arange((j - i) * bucketsize),
            np.arange(j - i + 1) * bucketsize,
            amp_and_duck(power_ds[i:j], quiet_target),
        )
        # dbg[i * bucketsize : j * bucketsize] /= amt
        output[i * bucketsize : j * bucketsize] *= np.c_[amt]
    out = WaveWrite(args.output)
    out.setparams(params._replace(nchannels=output.shape[1], framerate=samplerate))
    out.writeframes(output.reshape(-1).tobytes())
    out.close()


if __name__ == "__main__":
    main()
