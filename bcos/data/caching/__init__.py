# on-the-fly caching from bagua
from .cached_imagefolder import CachedImageFolder

# move to shm to cache
from .shm_caching import cache_tar_files_to_shm
