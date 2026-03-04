import sys
import os
import argparse

# add talebook root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from audiobook_generator.book_parsers.epub_book_parser import EpubBookParser
from audiobook_generator.config.general_config import GeneralConfig
import logging

logging.basicConfig(level=logging.INFO)

if len(sys.argv) > 1:
    dummy_args = argparse.Namespace(
        input_file=sys.argv[1],
        output_folder=".",
        preview=False,
        output_text=False,
        log='INFO',
        model_name="models/tts_models/en/vctk/vits",
        tts_kwargs=False,
        language="zh",
        voice_name="zh-CN-XiaoxiaoNeural",
        voice_rate="+0%",
        voice_volume="+0%",
        voice_pitch="+0%",
        newline_mode="none",
        chapter_start=1,
        chapter_end=-1,
        remove_endnotes=False,
        remove_reference_numbers=False,
        search_and_replace_file=None,
        title_mode="auto"
    )
    config = GeneralConfig(dummy_args)
    config.input_file = sys.argv[1]

    parser = EpubBookParser(config)
    parser.validate_config()

    chapters = parser.get_chapters("\n")
    print(f"Extracted {len(chapters)} chapters:")
    for i, (title, content) in enumerate(chapters[:3]):
        print(f"\n--- Chapter {i+1}: {title} ---")
        print(f"Content Length: {len(content)}")
        print(f"Preview: {content[:100]}...")
else:
    print("Please provide epub file path")
