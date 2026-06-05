from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from iox2_jsonrpc.iox2_transport import Iceoryx2JsonRpcServer
from common import EmptyParams, add_config_arg, build_processor, load_service_name


class PipelineBaseModel(BaseModel):
    service: Literal["serverA"] = "serverA"

    def model_post_init(self, context):
        print(f"[serverA {self.__class__.__name__}]")
        return super().model_post_init(context)
    
class StartPipelineParams(PipelineBaseModel):
    profile: str = Field(default="default")

class PipelineResult(PipelineBaseModel):
    running: bool
    profile: str


@dataclass
class PipelineController:
    running: bool = False
    profile: str = "default"

    def start(self, params: StartPipelineParams) -> PipelineResult:
        self.running = True
        self.profile = params.profile
        return PipelineResult(running=self.running, profile=self.profile)

    def stop(self, params: EmptyParams) -> PipelineResult:
        self.running = False
        return PipelineResult(running=self.running, profile=self.profile)

    def status(self, params: EmptyParams) -> PipelineResult:
        return PipelineResult(running=self.running, profile=self.profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="serverA: pipeline JSON-RPC service over iceoryx2")
    add_config_arg(parser)
    parser.add_argument("--service-key", default="serverA")
    args = parser.parse_args()

    iox_service_name = load_service_name(args.config, args.service_key)

    processor=build_processor(PipelineController(),prefix="pipeline")
    
    server = Iceoryx2JsonRpcServer(service_name=iox_service_name, processor=processor)
    server.serve_forever()


if __name__ == "__main__":
    main()
