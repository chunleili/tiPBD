# python convert.py -i input.yaml -o output.json
# python convert.py -i input.json -o output.yaml

import argparse
import json
import yaml
import os

def yaml_to_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Converted YAML {input_file} to JSON {output_file}.")

def json_to_yaml(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"Converted JSON {input_file} to YAML {output_file}.")
    
def main():
    parser = argparse.ArgumentParser(description="Convert YAML to JSON and JSON to YAML")
    parser.add_argument("-i", "--input", required=True, help="Input file (YAML or JSON)")
    parser.add_argument("-o", "--output", required=True, help="Output file (JSON or YAML)")
    args = parser.parse_args()

    in_ext = os.path.splitext(args.input)[1].lower()
    out_ext = os.path.splitext(args.output)[1].lower()

    if in_ext in ['.yaml', '.yml'] and out_ext == '.json':
        yaml_to_json(args.input, args.output)
    elif in_ext == '.json' and out_ext in ['.yaml', '.yml']:
        json_to_yaml(args.input, args.output)
    else:
        print("不支持的转换格式，确保输入和输出文件扩展名对应 YAML 或 JSON。")
        parser.print_help()

if __name__ == "__main__":
    main()