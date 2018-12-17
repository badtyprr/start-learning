# Image scrapers

# Python Packages
import os
import re
# 3rd Party Packages
import requests
import cv2
# User Packages
from .base import Scraper

class BingImageScraper(Scraper):
    def __init__(self,
                 subscription_key: str,
                 max_results: int=250,
                 group_size: int=50,
                 timeout: int=30,
                 min_width: int=0,
                 min_height: int=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscription_key = subscription_key
        self.max_results = max_results
        self.group_size = group_size
        self.timeout = timeout
        self.min_width = min_width
        self.min_height = min_height
        self.initializeAzure()

    def initializeAzure(self):
        self.search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

    def scrape(self, search_term: str, label: str):
        """
        Scrapes Bing Images for the search term and saves images to directory specified in label
        :param search_term: search term for Bing Images
        :param label: the label/directory name to save images to
        :return: Number of downloaded images
        """

        label_dir = os.path.join(self.output_dir, label.replace(' ', '_'))
        # Doesn't exist
        if not os.path.exists(label_dir):
            # Create label directory
            # Source: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
            os.makedirs(label_dir)
            print('[WARNING] Output directory did not exist, so it was created')
            N_start = 0
        # Does exist
        else:
            # Determine the image count in the label directory, and start at i+1
            files = os.listdir(label_dir)
            N_start = 0
            for file in files:
                # Matches image files that are in the format *[enumeration].[ext]
                # Thanks: https://javascript.info/regexp-greedy-and-lazy
                re_pattern = r'.*?(\d+)\..*'
                re_filenumber = re.compile(re_pattern, re.ASCII)
                filenumber = int(re_filenumber.fullmatch(file).group(1))
                N_start = max(N_start, filenumber)
        # Start on the next file
        N_start += 1
        # Continuously update N_end until the last image is downloaded
        N_end = N_start
        """
        # Possible exceptions
        """
        exception_set = [
            IOError, FileNotFoundError,
            requests.RequestException, requests.HTTPError,
            requests.ConnectionError, requests.Timeout,
            requests.exceptions.SSLError, requests.exceptions.ConnectTimeout,
            requests.TooManyRedirects, requests.exceptions.ContentDecodingError,
            requests.ChunkedEncodingError
        ]
        """
        # Possible image extensions
        """
        extensions = [
            'jpg', 'jpeg', 'jpe', 'png', 'bmp',
            'pbm', 'pgm', 'ppm', 'sr', 'ras',
            'jp2', 'tiff', 'tif'
        ]
        """
        # Search query
        """
        term = search_term
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        params = {"q": search_term, "imageType": "photo"}
        print("[INFO] searching Bing API for '{}'".format(term))

        try:
            response = requests.get(self.search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()

            estNumResults = min(search_results["totalEstimatedMatches"], self.max_results)
            print("[INFO] {} total results for '{}'".format(estNumResults, term))

            # Download images
            # Loop over total results in self.group_size batches
            for offset in range(0, estNumResults, self.group_size):
                print("[INFO] making request for group {}-{} of {}...".format(
                    offset,
                    offset + self.group_size,
                    estNumResults
                ))
                params["offset"] = offset
                search = requests.get(self.search_url, headers=headers, params=params)
                search.raise_for_status()
                results = search.json()
                print("[INFO] saving images for group {}-{} of {}...".format(
                    offset,
                    offset + self.group_size,
                    estNumResults
                ))
                # Loop over group batch
                for v in results["value"]:
                    try:
                        print("[INFO] fetching: {}".format(v["contentUrl"]))
                        try:
                            r = requests.get(v["contentUrl"], timeout=self.timeout)
                        except requests.ReadTimeout:
                            continue
                        ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                        ext = ext.lower()

                        if v["width"] < self.min_width or v["height"] < self.min_height:
                            print("[WARNING] {}x{} is smaller than {}x{}, skipping: {}".format(
                                v["width"], v["height"],
                                self.min_width, self.min_height,
                                v["contentUrl"])
                            )
                            continue
                        # Check for valid extensions
                        elif ext[1:] not in extensions:

                            ext = ext[:4]
                            if ext[1:] not in extensions:
                                # default to .jpg
                                ext = '.jpg'
                                print("[WARNING] invalid extension, assigning extension as jpg: {}".format(v["contentUrl"]))
                            else:
                                print("[WARNING] invalid extension, truncating extension: {}".format(v["contentUrl"]))
                        # path in format 'outputPath/number.ext'
                        p = os.path.sep.join(
                            [label_dir,
                             "{}{}".format(str(N_end).zfill(8), ext)]
                        )
                        # Write to disk
                        f = open(p, "wb")
                        f.write(r.content)
                        f.close()
                    except Exception as e:
                        if type(e) in exception_set:
                            print("[WARNING] encountered exception: {}, skipping: {}".format(e, v["contentUrl"]))
                            continue
                        else:
                            raise e
                    """
                    # verify it's a good image
                    """
                    image = cv2.imread(p)
                    if image is None:
                        print("[WARNING] deleting corrupted image: {}".format(p))
                        os.remove(p)
                    # Found a good image, go to the next
                    N_end += 1
        except requests.HTTPError as e:
            # Sometimes you'll get this error if you entered the wrong subscription key
            # Source: https://blogs.msdn.microsoft.com/kwill/2017/05/17/http-401-access-denied-when-calling-azure-cognitive-services-apis/
            print(e)
        # Return the end index less start index
        return N_end-N_start+1

