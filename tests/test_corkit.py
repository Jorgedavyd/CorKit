import os
import shutil
from datetime import datetime
import asyncio
from ..corkit.lasco import CME, level_1, downloader

async def tool_downloader(tool: str) -> None:

    down = downloader(tool, ".test")
    await down(scrap_date_list)

def test_downloader() -> None:
    scrap_date_list = [
        (datetime(1998, 5, 6), datetime(1998, 5, 7)),  # Solar Storm of May 1998
    ]


    tools = ["c2", "c3"]
    ## Downloader

    os.makedirs("test", exist_ok=True)


    asyncio.run(*[tool_downloader(tool) for tool in tools])

    ## Checking

    path = lambda name: f"test/{name}"

    for name in tools:
        file_list = [
            os.path.join(path(name), filename)
            for filename in sorted(os.listdir(path(name)))
        ]
        level_1(file_list, path(name))


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


    shutil.rmtree("./test")
    print("CME analysis test done!")
    "CME analysis test done!")
