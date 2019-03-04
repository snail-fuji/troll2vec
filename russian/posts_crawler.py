import scrapy
import pdb
from tqdm import tqdm

PAGES = 1000

class PikabuSpider(scrapy.Spider):
  name = 'pikabu'
  start_urls = ['https://pikabu.ru/new?page={}'.format(i) for i in range(PAGES)]

  def parse(self, response):
    for post in response.css('.stories-feed__container > .story'):
      yield {
        'url': post.css('a.story__title-link ::attr(href)').get(),
        'title': post.css('a.story__title-link ::text').get(),
        'tags': post.css('a.tags__tag ::text').extract(),
      }

