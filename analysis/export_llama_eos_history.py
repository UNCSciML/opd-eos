#!/usr/bin/env python3
"""Export selected public run metrics; credentials stay in memory."""
import argparse
import getpass
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    api = wandb.Api(api_key=getpass.getpass('W&B credential (hidden): '), timeout=60)
    run = api.run(args.run)
    selected = {'training/global_step', '_step', 'response_length/mean',
                'finish_reason/max_tokens_fraction'}
    prefixes = ('student_eos_prob/', 'teacher_eos_prob/', 'finish_reason/')
    history = [{k: v for k, v in row.items() if k in selected or k.startswith(prefixes)}
               for row in run.scan_history(page_size=500)]
    # Retain model/training settings, never environment variables or credentials.
    config = run.config
    def at(*keys):
        value = config
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value
    metadata = {
        'run_path': args.run, 'name': run.name, 'state': run.state, 'url': run.url,
        'config': {
            'student': at('actor_rollout_ref', 'model', 'path'),
            'teacher': at('reward_model', 'model', 'path'),
            'batch_size': at('data', 'train_batch_size'),
            'max_prompt_length': at('data', 'max_prompt_length'),
            'max_response_length': at('data', 'max_response_length'),
            'train_files': at('data', 'train_files'),
            'chat_template_model': at('data', 'chat_template_model'),
            'apply_chat_template_kwargs': at('data', 'apply_chat_template_kwargs'),
            'rollout_n': at('actor_rollout_ref', 'rollout', 'n'),
            'temperature': at('actor_rollout_ref', 'rollout', 'temperature'),
            'eos_mode': at('actor_rollout_ref', 'rollout', 'eos_mode'),
            'semantic_eos_token_ids': at('actor_rollout_ref', 'rollout', 'semantic_eos_token_ids'),
            'total_training_steps': at('trainer', 'total_training_steps'),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    steps = [r['training/global_step'] for r in history if 'training/global_step' in r]
    print('history rows:', len(history), 'training steps:', len(steps), 'range:', min(steps), max(steps))


if __name__ == '__main__':
    main()
