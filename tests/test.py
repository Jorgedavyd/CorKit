import os
import shutil
from datetime import datetime
import asyncio

def test() -> None:
    scrap_date_list = [
        (datetime(1998, 5, 6), datetime(1998, 5, 7)),  # Solar Storm of May 1998
    ]


    tools = ["c2", "c3"]
    ## Downloader
    from corkit.lasco import downloader

    os.makedirs("test", exist_ok=True)


    async def test_downloader(tool: str) -> None:
        down = downloader(tool, "./test")
        await down(scrap_date_list)


    for tool in tools:
        asyncio.run(test_downloader(tool))

    ## Checking
    from corkit.lasco import level_1

    path = lambda name: f"test/{name}"

    for name in tools:
        file_list = [
            os.path.join(path(name), filename)
            for filename in sorted(os.listdir(path(name)))
        ]
        level_1(file_list, path(name))

    print("Level 1 calibration test done!")

    from corkit.lasco import CME

    cme = CME()

    bn = lambda name: os.path.join(f"test/{name}", sorted(os.listdir(f"./test/{name}"))[0])

    for name in tools:
        cme.mass(
            bn(name),
            [
                os.path.join(path(name), filename)
                for filename in sorted(os.listdir(path(name)))
            ],
            ALL=True,
        )

    print("CME analysis test done!")

    shutil.rmtree("./test")
