import os
import sys
import math
import subprocess
import argparse

def split(path, nr_thread, has_header):

    def open_with_header_witten(path, idx, header):
        f = open(path+'.__tmp__.{0}'.format(idx), 'w')
        if not has_header:
            return f
        f.write(header)
        return f

    def open_with_first_line_skipped(path, skip=True):
        f = open(path)
        if not skip:
            return f
        next(f)
        return f

    def calc_nr_lines_per_thread():
        nr_lines = int(list(subprocess.Popen('wc -l {0}'.format(path), shell=True,
            stdout=subprocess.PIPE).stdout)[0].split()[0])
        if not has_header:
            nr_lines += 1
        return math.ceil(float(nr_lines)/nr_thread)

    header = open(path).readline()

    nr_lines_per_thread = calc_nr_lines_per_thread()

    idx = 0
    f = open_with_header_witten(path, idx, header)
    for i, line in enumerate(open_with_first_line_skipped(path, has_header), start=1):
        if i%nr_lines_per_thread == 0:
            f.close()
            idx += 1
            f = open_with_header_witten(path, idx, header)
        f.write(line)
    f.close()


def parallel_convert(cvt_path, arg_paths, nr_thread):
    workers = []
    for i in range(nr_thread):
        cmd = 'python {0}'.format(os.path.join('.', cvt_path))
        for path in arg_paths:
            cmd += ' {0}'.format(path+'.__tmp__.{0}'.format(i))
        worker = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        workers.append(worker)
    for worker in workers:
        worker.communicate()


def delete(path, nr_thread):
    for i in range(nr_thread):
        os.remove('{0}.__tmp__.{1}'.format(path, i))


def cat(path, nr_thread):
    if os.path.exists(path):
        os.remove(path)
    for i in range(nr_thread):
        cmd = 'cat {svm}.__tmp__.{idx} >> {svm}'.format(svm=path, idx=i)
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()


def parse_args():
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='nr_thread', default=12, type=int)
    parser.add_argument('cvt_path')
    parser.add_argument('src_path')
    parser.add_argument('dst_path')
    args = vars(parser.parse_args())

    return args

def main():
    args = parse_args()

    nr_thread = args['nr_thread']

    split(args['src_path'], nr_thread, True)

    parallel_convert(args['cvt_path'], [args['src_path'], args['dst_path']], nr_thread)

    delete(args['src_path'], nr_thread)

    cat(args['dst_path'], nr_thread)

    delete(args['dst_path'], nr_thread)

main()
