#!/usr/bin/env python
import argparse
import json
import logging
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

from ms_converter._version import __version__
from .device_code import replace_device_code
from .extras import remove_extra_code
from .param_code import replace_param_code
from .temp_code import replace_temporary_code
MINT_API_TABLE = {
    "2.5.0": "mindspore_v2.5.0.mint.rst",
    "2.6.0": "mindspore_v2.6.0.mint.rst"
}

ASSETS_DIR = os.path.join(sys.prefix, "ms_converter/assets")
DEFAULT_MAPPING = os.path.join(ASSETS_DIR, "mapping.json")

_logger = logging.getLogger(__name__)


def scan_mint_api(ms_version: str = "2.5",
                  mint_api_path: Optional[str] = None) -> List[str]:
    if mint_api_path is not None:
        if not mint_api_path.endswith(".rst"):
            raise ValueError(
                "`mint_api_path` shoud be a path ends with `.rst`.")
        mint_api_path = os.path.abspath(mint_api_path)
        _logger.debug("Reading the custom MindSpore Mint API doc from %s",
                      mint_api_path)
    else:
        mint_api_path = os.path.abspath(
            os.path.join(ASSETS_DIR, MINT_API_TABLE[ms_version]))
        _logger.debug("Reading the default MindSpore Mint API doc from %s",
                      mint_api_path)

    with open(mint_api_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = re.findall(r"[ +]mindspore.mint.(\w.+)", content)
    _logger.debug("Collected %d Mint APIs", len(result))
    return result


def form_base_torch_mint_mapping(api: List[str]) -> Dict[str, str]:
    mapping = dict()
    for x in api:
        torch_api_name = "torch." + x
        mint_api_name = "mint." + x
        mapping[torch_api_name] = mint_api_name
    return mapping


def is_inclusive(mapping: Dict[str, str]) -> bool:
    non_inclusive_list = list()
    for x in mapping.keys():
        for v in non_inclusive_list:
            if v in x:
                _logger.warning(
                    "%s is already in the replacing list with key %s", v, x)
                return True
        non_inclusive_list.append(x)
    return False


def expand_torch_mint_mapping(
        mapping: Dict[str, str],
        custom_mapping_path: Optional[str] = None) -> None:
    extra_mapping = os.path.abspath(DEFAULT_MAPPING)
    _logger.debug("Reading the mapping from %s", extra_mapping)
    with open(extra_mapping, "r") as f:
        extra_mapping = json.load(f)

    if custom_mapping_path is not None:
        custom_mapping_path = os.path.abspath(custom_mapping_path)
        _logger.debug("Reading the custom mapping from %s",
                      custom_mapping_path)
        with open(custom_mapping_path, "r") as f:
            custom_mapping = json.load(f)
        extra_mapping.update(custom_mapping)

    if is_inclusive(extra_mapping):
        raise ValueError(
            "There is some partially-duplicated keys in the mapping list. "
            "We did not support this kind of keys yet.")

    # in pytorch script, we usually use nn.xxx instead of torch.nn.xxx
    for u, v in mapping.items():
        if ".nn." in u:
            extra_mapping[u.replace("torch.", "")] = v

    mapping.update(extra_mapping)
    return


def _process_single_file(
    input_file: str,
    ms_version: str = "2.5",
    mint_api_path: Optional[str] = None,
    custom_mapping_path: Optional[str] = None,
    replace: bool = False,
    extra_conversion: bool = False,
    temp_conversion: bool = False
) -> None:
    """Process a single Python file for torch to mint conversion."""
    # Read the file first to check if it contains torch import
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if file contains torch import
    if "import torch" not in content:
        _logger.debug(f"Skipping {input_file} - no torch import found")
        return

    _logger.info(f"Converting {input_file}")
    input_file = os.path.abspath(input_file)
    api = scan_mint_api(ms_version=ms_version, mint_api_path=mint_api_path)
    mapping = form_base_torch_mint_mapping(api)
    expand_torch_mint_mapping(mapping, custom_mapping_path=custom_mapping_path)

    content = replace_device_code(content)
    content = replace_param_code(content)
    if extra_conversion:
        content = remove_extra_code(content)

    for u, v in mapping.items():
        if u not in content:
            continue

        content = re.sub(u, v, content)
        _logger.debug(f"Update: {u.strip():<40} --> {v.strip():<40}".replace(
            "\n", " "))

    if temp_conversion:
        content = replace_temporary_code(content)
    if replace:
        backup = input_file + ".old"
        shutil.move(input_file, backup)
        with open(input_file, "w") as f:
            f.write(content)
        _logger.debug(
            "Replacement is finished. Old file is saved as %s",
            backup)
    else:
        print(f"\n=== Converted content for {input_file} ===")
        print(content)
        print("=" * 80)


def _torch2mint(
    input_: str,
    ms_version: str = "2.5",
    mint_api_path: Optional[str] = None,
    custom_mapping_path: Optional[str] = None,
    replace: bool = False,
    extra_conversion: bool = False,
    temp_conversion: bool = False
) -> None:
    input_ = os.path.abspath(input_)
    
    if os.path.isfile(input_):
        _process_single_file(
            input_,
            ms_version=ms_version,
            mint_api_path=mint_api_path,
            custom_mapping_path=custom_mapping_path,
            replace=replace,
            extra_conversion=extra_conversion,
            temp_conversion=temp_conversion
        )
    elif os.path.isdir(input_):
        _logger.info(f"Processing directory: {input_}")
        for root, _, files in os.walk(input_):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        _process_single_file(
                            file_path,
                            ms_version=ms_version,
                            mint_api_path=mint_api_path,
                            custom_mapping_path=custom_mapping_path,
                            replace=replace,
                            extra_conversion=extra_conversion,
                            temp_conversion=temp_conversion
                        )
                    except Exception as e:
                        _logger.error(f"Error processing {file_path}: {str(e)}")
    else:
        raise ValueError(f"Input path {input_} is neither a file nor a directory")


def main():
    parser = argparse.ArgumentParser(
        usage=
        "Convert script(s) from using PyTorch API to MindSpore API (mint API) partially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument("-i",
                        "--input", help="Path of the script or directory to be converted.")
    parser.add_argument("--replace",
                        action="store_true",
                        help="make the update to files in place")
    parser.add_argument("--mint-api-path", help="Path to the Mint API list")
    parser.add_argument("-c",
                        "--custom-mapping-path",
                        help="Path to the custom mapping list")
    parser.add_argument("--ms-version",
                        choices=["2.5.0", "2.6.0"],
                        default="2.5.0",
                        help="MindSpore version for API mapping.")
    parser.add_argument("-vv",
                        "--verbose",
                        action="store_true",
                        help="Shown the debug message")
    parser.add_argument("-e",
                        "--extra_conversion",
                        action="store_true",
                        help="run extra conversion with ast parse")
    parser.add_argument("-t",
                        "--temp_conversion",
                        action="store_true",
                        help="run temporary conversion with fixed rules")
    args = parser.parse_args()

    # set logger
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    _logger.setLevel(logging_level)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    _logger.addHandler(console_handler)

    _torch2mint(
        args.input,
        ms_version=args.ms_version,
        mint_api_path=args.mint_api_path,
        custom_mapping_path=args.custom_mapping_path,
        replace=args.replace,
        extra_conversion=args.extra_conversion,
        temp_conversion=args.temp_conversion,
    )


if __name__ == "__main__":
    main()
