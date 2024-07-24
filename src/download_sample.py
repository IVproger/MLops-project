from pathlib import Path

import gdown

Path(".dvc/datastore/files/md5/5e").mkdir(parents=True, exist_ok=True)
gdown.download(
    "https://drive.google.com/uc?id=1OrOPS0GDsFno3FzukmOqPzETFsDYbiiU",
    ".dvc/datastore/files/md5/5e/e07f6e03b526e9164699b19e5036d9",
    quiet=False,
    use_cookies=False,
)
