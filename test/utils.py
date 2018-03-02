#encoding=utf-8
import re
import requests
import requests.exceptions as reExceptions
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def changeRawToJson(str):
    s1 = "{\""
    str = str.replace("=","\":\"")
    str = str.replace("&",'","')
    return "{\"" + str + "\"}"

def changeRawToForm(str):
    str = str.replace("=",":")
    str = str.replace("&","\n")
    str = str.replace("%20"," ")
    str = str.replace("%2F","/")
    return str

def changeRawToDict(str):
    str = changeRawToForm(str)
    str = str.split('\n')
    array = {}
    for item in str:
        item = item.split(':')
        array[item[0]] = item[1]
    return array

def translateFormData(str):
    str = str.replace("%20"," ")
    str = str.replace("%2F","/")
    return str

def getPicUrl(html):
    try:
        result = requests.get(html)
        result.encoding = "gb2312"
        target_url = re.findall(pattern='.html\'\s*><img alt=.*?"\s*src="(.*?)"\s*/></a>' ,string=result.text)
    except reExceptions.ConnectionError:
        print("connection Error:" + html)
        target_url = ""
    if len(target_url) > 0:
        return target_url[0]
    return ""

def resizeImage(fileName,shape,newFileName):
    if os.path.exists(fileName) == False:
        return
    image = Image.open(fileName)
    image.thumbnail((shape[0], shape[1]), Image.ANTIALIAS)
    if os.path.exists(newFileName):
        writer = open(newFileName,'w+b')
        writer.close()
    image.save(newFileName, 'JPEG', quality=shape[2])


