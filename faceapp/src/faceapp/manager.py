import asyncio
import threading
from queue import Queue

from faceapp._base.pipeline import Pipeline


class PipelineManager:
    def __init__(self, producer_pipeline: Pipeline, consumer_pipeline: Pipeline):
        self.producer_pipeline = producer_pipeline
        self.consumer_pipeline = consumer_pipeline
        self.queue = Queue()

    async def run(self, producer_config: dict = None, consumer_config: dict = None):
        # start producer and consumer in two threads
        t1 = threading.Thread(
            target=self.start_loop,
            args=(self.producer(producer_config),),
            name="producer",
        )
        t2 = threading.Thread(
            target=self.start_loop,
            args=(self.consumer(consumer_config),),
            name="consumer",
        )

        t1.start()
        t2.start()
        t1.join()
        t2.join()

    async def producer(self, config: dict):
        """Producer that pushes items into the queue"""
        if not config:
            config = {}
        c = 0
        async for extraction in self.producer_pipeline.ainvoke(**config):
            self.queue.put(extraction)
            c += 1
            print("Processed image count:", c)
        self.queue.put(None)

    async def consumer(self, config: dict):
        """Consumer that reads items from the queue"""
        if not config:
            config = {}
        while True:
            item = self.queue.get()
            if item is None:
                break

            output = await self.consumer_pipeline.ainvoke(**item, **config)
            print(output)
        self.queue.task_done()

    @staticmethod
    def start_loop(coro):
        # TODO: Investigate true event loop behaviour or not
        asyncio.run(coro)
