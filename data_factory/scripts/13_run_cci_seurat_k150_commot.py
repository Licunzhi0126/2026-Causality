#!/usr/bin/env python3
from __future__ import annotations

from run_cci_layer_commot import build_argparser, run_layer


def main() -> None:
    run_layer(build_argparser(default_layer="seurat_k150").parse_args())


if __name__ == "__main__":
    main()
