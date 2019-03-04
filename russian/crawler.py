import scrapy
import pdb
from tqdm import tqdm

PAGES = 1000

class PikabuSpider(scrapy.Spider):
  name = 'pikabu'
  start_urls = ['https://pikabu.ru/new?page={}'.format(i) for i in range(PAGES)]

  def parse(self, response):
    for next_page in response.css('.story__title a'):
      yield response.follow(next_page, self.parse_story)

  def parse_story(self, response):
    for comment in response.css('.page-story__comments .comment'):
      body = comment.css(".comment__body:first-child")
      yield {
        'url': response.url,
        'rating': body.css('.comment__rating-count ::attr(aria-label)').get(),
        'author': body.css('.user__nick ::text').get(),
        'comment': body.css('.comment__content').get(),
        'meta': comment.css('::attr(data-meta)').get(),
      }
