'''Input module for reading pre-processed data in.'''
import struct
import sys
import numpy


def get_input(file_object):
    '''Read pre-processed data in.
    `file_object` should be a file object, which supports binary read.
    Data will be started reading from the current position of the file.
    This is a generator function; each of its iterated object is a tuple like
    (song_id, sample_count, fft_data, timing_data).
    '''
    while True:
        song_id = file_object.read(4)
        if song_id == b'':
            return
        song_id = struct.unpack('=i', song_id)[0]
        sample_count = struct.unpack('=i', file_object.read(4))[0]
        fft_data = numpy.array(list(struct.iter_unpack(
            '=' + 'f' * 64, file_object.read(4 * 64 * sample_count))))
        timing_data = numpy.array(list(struct.iter_unpack(
            '=b', file_object.read(sample_count))))
        yield song_id, sample_count, fft_data, timing_data[:, 0]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('There must be only one parameter for the input file.')
        sys.exit(0)
    with open(sys.argv[1], 'rb') as fp:
        for song, samples, fft, timing in get_input(fp):
            print('Song #{}: {} samples, FFT data extent {}, timing data extent {}.'.format(
                song, samples, fft.shape, timing.shape))
