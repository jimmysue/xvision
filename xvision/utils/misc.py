import os
import math
import multiprocessing

# copy and edit from https://github.com/conan-io/conan/blob/develop/conans/client/tools/oss.py

class CpuProperties(object):

    def get_cpu_quota(self):
        return int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").readlines())

    def get_cpu_period(self):
        return int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").readlines())

    def get_cpus(self):
        try:
            cfs_quota_us = self.get_cpu_quota()
            cfs_period_us = self.get_cpu_period()
            if cfs_quota_us > 0 and cfs_period_us > 0:
                return int(math.ceil(cfs_quota_us / cfs_period_us))
        except:
            pass
        return multiprocessing.cpu_count()


def cpu_count(output=None):
    try:
        env_cpu_count = os.getenv("CONAN_CPU_COUNT", None)
        if env_cpu_count is not None and not env_cpu_count.isdigit():
            raise RuntimeError("Invalid CONAN_CPU_COUNT value '%s', "
                                 "please specify a positive integer" % env_cpu_count)
        if env_cpu_count:
            return int(env_cpu_count)
        else:
            return CpuProperties().get_cpus()
    except NotImplementedError:
        output.warn(
            "multiprocessing.cpu_count() not implemented. Defaulting to 1 cpu")
    return 1  # Safe guess


if __name__ == '__main__':

    n = cpu_count()

    print(n)