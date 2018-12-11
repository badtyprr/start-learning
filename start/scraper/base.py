# Base Scraper class

# Python Packages
import os
from abc import ABC, abstractmethod

class Scraper(ABC):
    def __init__(self, output_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        # Create output directory
        # Source: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('[WARNING] Output directory did not exist, so it was created')


    @abstractmethod
    def scrape(self, search_term: str, label: str):
        pass