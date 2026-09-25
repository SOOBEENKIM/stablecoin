import dv_paths as original
from dv_alignment import aligned_read,verify_alignment


if __name__=='__main__':
    verify_alignment()
    original.read=aligned_read
    original.main()
