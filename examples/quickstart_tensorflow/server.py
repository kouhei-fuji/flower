import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def main(args):
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-address",
        default="0.0.0.0:8080",
        help="<ip address>:<port> ex) 192.168.0.10:50051",
    )
    parser.add_argument("--num-rounds", type=int, default=3)
    main(parser.parse_args())
