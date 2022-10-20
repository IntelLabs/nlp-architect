# ******************************************************************************
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Driver script to start and run InterpreT.

This script will start InterpreT on the local machine on port 5005
(or another port which is specificied by command line arguments).
"""

import os
import socket
import logging
import argparse

import appConfiguration
from tasks import get_task


def main(port):
    collateral = os.getenv("COLLATERAL").split(",")
    df = os.getenv("DF").split(",")
    name = os.getenv("NAME").split(",")
    task = os.getenv("TASK")
    num_layers = int(os.getenv("NUM_LAYERS"))

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    model_names = name if name and len(name) == len(collateral) else ""
    Task = get_task(task, num_layers, model_names, collateral, df)
    app = appConfiguration.configureApp(Task)
    appConfiguration.printPageLink(hostname, port)
    app.server.run(debug=True, threaded=True, host=ip_address, port=int(port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collateral", nargs="*", help="Path to collateral (.pt/.pkl)")
    parser.add_argument("--df", nargs="*", help="Path to dataframe (.csv/.dat)")
    parser.add_argument("--name", nargs="*", help="Model names to display")
    parser.add_argument("--port", default="5005", type=str, help="port")
    parser.add_argument(
        "--task", default="wsc", type=str, help="absa or wsc", choices=["wsc", "absa"]
    )
    parser.add_argument("--num_layers", type=str, help="Number of transformer layers")

    args = parser.parse_args()

    assert len(args.collateral) > 0, "Collateral path not provided"
    assert len(args.df) > 0, "Csv path not provided"
    if not args.num_layers:
        logging.warning("--num_layers not provided, defaulting to 12...")
        args.num_layers = "12"

    os.environ["COLLATERAL"] = ",".join(args.collateral)
    os.environ["DF"] = ",".join(args.df)
    os.environ["NAME"] = ",".join(args.name)
    os.environ["TASK"] = args.task
    os.environ["NUM_LAYERS"] = args.num_layers

    main(args.port)
