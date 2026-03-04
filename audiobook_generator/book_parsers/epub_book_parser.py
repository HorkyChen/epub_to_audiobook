import logging
import re
from typing import List, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from audiobook_generator.book_parsers.base_book_parser import BaseBookParser
from audiobook_generator.config.general_config import GeneralConfig

logger = logging.getLogger(__name__)


class EpubBookParser(BaseBookParser):
    def __init__(self, config: GeneralConfig):
        super().__init__(config)
        self.book = epub.read_epub(self.config.input_file, {"ignore_ncx": True})

    def __str__(self) -> str:
        return super().__str__()

    def validate_config(self):
        if self.config.input_file is None:
            raise ValueError("Epub Parser: Input file cannot be empty")
        if not self.config.input_file.endswith(".epub"):
            raise ValueError(f"Epub Parser: Unsupported file format: {self.config.input_file}")

    def get_book(self):
        return self.book

    def get_book_title(self) -> str:
        if self.book.get_metadata('DC', 'title'):
            return self.book.get_metadata("DC", "title")[0][0]
        return "Untitled"

    def get_book_author(self) -> str:
        if self.book.get_metadata('DC', 'creator'):
            return self.book.get_metadata("DC", "creator")[0][0]
        return "Unknown"

    def get_chapters(self, break_string) -> List[Tuple[str, str]]:
        chapters = self.get_chapters_normal(break_string)
        # Check if only one chapter and the content is larger than 100K bytes
        if len(chapters) == 1 and len(chapters[0][1].encode('utf-8')) > 100000:
            logger.info("Single large chapter detected, attempting to extract with TOC...")
            toc_chapters = self.get_chapters_with_toc(break_string)
            if toc_chapters and len(toc_chapters) > 0:
                return toc_chapters
        return chapters

    def get_chapters_normal(self, break_string) -> List[Tuple[str, str]]:
        chapters = []
        search_and_replaces = self.get_search_and_replaces()
        for item_id, linear in self.book.spine:
            if linear == 'no' and item_id != 'nav':
                continue

            item = self.book.get_item_with_id(item_id)
            if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            content = item.get_content()
            soup = BeautifulSoup(content, "lxml-xml")
            raw = soup.get_text(strip=False)
            # logger.debug(f"Raw text: <{raw[:200]}>")

            # Replace excessive whitespaces and newline characters based on the mode
            if self.config.newline_mode == "single":
                cleaned_text = re.sub(r"[\n]+", break_string, raw.strip())
            elif self.config.newline_mode == "double":
                cleaned_text = re.sub(r"[\n]{2,}", break_string, raw.strip())
            elif self.config.newline_mode == "none":
                cleaned_text = re.sub(r"[\n]+", " ", raw.strip())
            else:
                raise ValueError(f"Invalid newline mode: {self.config.newline_mode}")

            logger.debug(f"Cleaned text step 1: <{cleaned_text[:]}>")
            cleaned_text = re.sub(r"\s+", " ", cleaned_text)
            logger.debug(f"Cleaned text step 2: <{cleaned_text[:100]}>")

            # Removes end-note numbers
            if self.config.remove_endnotes:
                cleaned_text = re.sub(r'(?<=[a-zA-Z.,!?;”")])\d+', "", cleaned_text)
                logger.debug(f"Cleaned text step 4: <{cleaned_text[:100]}>")

            # Removes references numbers like [1] or [2.3]
            if self.config.remove_reference_numbers:
                cleaned_text = re.sub(r'\[\d+(\.\d+)?\]', '', cleaned_text)
                logger.debug(f"Cleaned text step 4.1 (removed brackets): <{cleaned_text[:100]}>")

            # Does user defined search and replaces
            for search_and_replace in search_and_replaces:
                cleaned_text = re.sub(search_and_replace['search'], search_and_replace['replace'], cleaned_text)
            logger.debug(f"Cleaned text step 5: <{cleaned_text[:100]}>")

            # Get proper chapter title
            if self.config.title_mode == "auto":
                title = ""
                title_levels = ['title', 'h1', 'h2', 'h3']
                for level in title_levels:
                    if soup.find(level):
                        title = soup.find(level).text
                        break
                if title.strip() == "" or re.match(r'^\d{1,3}$',title) is not None:
                    title = cleaned_text[:60]
            elif self.config.title_mode == "tag_text":
                title = ""
                title_levels = ['title', 'h1', 'h2', 'h3']
                for level in title_levels:
                    if soup.find(level):
                        title = soup.find(level).text
                        break
                if title.strip() == "":
                    title = "<blank>"
            elif self.config.title_mode == "first_few":
                title = cleaned_text[:60]
            else:
                raise ValueError("Unsupported title_mode")
            logger.debug(f"Raw title: <{title}>")
            title = self._sanitize_title(title, break_string)
            logger.debug(f"Sanitized title: <{title}>")

            chapters.append((title, cleaned_text))
            soup.decompose()
        return chapters

    def get_chapters_with_toc(self, break_string) -> List[Tuple[str, str]]:
        from bs4 import NavigableString

        def _extract_toc_items(toc_elements):
            items = []
            for entry in toc_elements:
                if isinstance(entry, epub.Link):
                    items.append((entry.title, entry.href))
                elif isinstance(entry, epub.Section):
                    items.append((entry.title, entry.href))
                    if entry.children:
                        items.extend(_extract_toc_items(entry.children))
                elif isinstance(entry, tuple) and len(entry) == 2:
                    section, children = entry
                    if hasattr(section, "title") and hasattr(section, "href"):
                        items.append((section.title, section.href))
                    else:
                        items.append((str(section), None))
                    if children:
                        items.extend(_extract_toc_items(children))
                else:
                    logger.warning(f"Unknown TOC entry type: {entry}")
            return items

        toc_items = _extract_toc_items(self.book.toc)
        logger.info(f"Discovered {len(toc_items)} TOC items")
        if not toc_items:
            return []

        first_href = toc_items[0][1]
        if not first_href:
            return []

        file_name = first_href.split("#")[0]
        item = self.book.get_item_with_href(file_name)
        if item is None:
            for it in self.book.get_items():
                if it.get_name().endswith(file_name):
                    item = it
                    break
        if item is None:
            logger.error(f"Failed to find file {file_name} from TOC in book items")
            return []

        content = item.get_content().decode("utf-8")
        soup = BeautifulSoup(content, "lxml-xml" if content.startswith("<?xml") else "html.parser")

        chapters_list = []
        for title, href in toc_items:
            frag = href.split("#")[-1] if "#" in href else None
            if not frag:
                continue

            elems = soup.select(f'[id="{frag}"], [name="{frag}"]')
            if elems:
                chapters_list.append((title, elems[0]))

        chapters_text = []
        search_and_replaces = self.get_search_and_replaces()
        for i, (current_title, current_elem) in enumerate(chapters_list):
            next_anchor = chapters_list[i + 1][1] if i + 1 < len(chapters_list) else None

            text_parts = []
            for elem in current_elem.next_elements:
                if elem == next_anchor:
                    break
                if isinstance(elem, NavigableString):
                    text_parts.append(str(elem))

            raw = "".join(text_parts)

            # Clean text applying same rules as get_chapters_normal
            if self.config.newline_mode == "single":
                cleaned_text = re.sub(r"[\n]+", break_string, raw.strip())
            elif self.config.newline_mode == "double":
                cleaned_text = re.sub(r"[\n]{2,}", break_string, raw.strip())
            elif self.config.newline_mode == "none":
                cleaned_text = re.sub(r"[\n]+", " ", raw.strip())
            else:
                raise ValueError(f"Invalid newline mode: {self.config.newline_mode}")

            cleaned_text = re.sub(r"\s+", " ", cleaned_text)

            if self.config.remove_endnotes:
                cleaned_text = re.sub(r'(?<=[a-zA-Z.,!?;”")])\d+', "", cleaned_text)
            if self.config.remove_reference_numbers:
                cleaned_text = re.sub(r'\[\d+(\.\d+)?\]', '', cleaned_text)

            for search_and_replace in search_and_replaces:
                cleaned_text = re.sub(search_and_replace['search'], search_and_replace['replace'], cleaned_text)

            sanitized_title = self._sanitize_title(current_title, break_string)
            chapters_text.append((sanitized_title, cleaned_text))

        return chapters_text

    def get_search_and_replaces(self):
        search_and_replaces = []
        if self.config.search_and_replace_file:
            with open(self.config.search_and_replace_file) as fp:
                search_and_replace_content = fp.readlines()
                for search_and_replace in search_and_replace_content:
                    if '==' in search_and_replace and not search_and_replace.startswith('==') and not search_and_replace.endswith('==') and not search_and_replace.startswith('#'):
                        search_and_replaces = search_and_replaces + [ {'search': r"{}".format(search_and_replace.split('==')[0]), 'replace': r"{}".format(search_and_replace.split('==')[1][:-1])} ]
        return search_and_replaces

    @staticmethod
    def _sanitize_title(title, break_string) -> str:
        # replace MAGIC_BREAK_STRING with a blank space
        # strip incase leading bank is missing
        title = title.replace(break_string, " ")
        sanitized_title = re.sub(r"[^\w\s]", "", title, flags=re.UNICODE)
        sanitized_title = re.sub(r"\s+", "_", sanitized_title.strip())
        return sanitized_title
