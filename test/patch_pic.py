#encoding=utf-8
import re
import requests
import utils as util
import urllib
import os
from bs4 import BeautifulSoup as bs


def patch_pic(base_url,filePath,encoding="gb2312"):
    # str = re.findall('<ul class="w110 oh Tag_list" id="Tag_list">(.*?)</ul>',str)
    if os.path.exists(filePath) == False:
        os.mkdir(filePath)

    html1 = requests.get(base_url)
    html1.encoding = encoding
    html1_text = html1.text
    html1_text = html1_text.split('<ul class="w110 oh Tag_list" id="Tag_list">')[-1].split('  </ul>')[0]
    html_list = re.findall('<li><a href="(.*?)" title="',html1_text)
    for i in range(0,len(html_list)):
        html_detail = html_list[i]
        index = html_detail.split('/')[-1].split('.html')[0]
        for j in range(1,6):
            pic_html = html_detail.replace(index,str(index) + "_" + str(j+1))
            pic_url = util.getPicUrl(pic_html)
            if pic_url == "":
                print("error url:" + pic_html)
                continue
            urllib.urlretrieve(pic_url, filePath + str(i* 5 + j) + ".jpg")

def resizePic(basePath):
    list = os.listdir(basePath)
    if os.path.exists(basePath + "reszie/") == False:
        os.mkdir(basePath + "reszie/")
    for file in list:
        if file == '.DS_Store':
            continue
        util.resizeImage(basePath + file,[256,256,120],newFileName=basePath + file)

resizePic("/Users/aria/MyDocs/pics/6_冷艳/")


