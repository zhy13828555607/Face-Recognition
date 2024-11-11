"""
Get the original data from https://image.baidu.com/
"""

from fake_useragent import UserAgent
import requests
import re
import uuid

headers = {"User-agent": UserAgent().random,
           "Accept-Encoding": "gzip, deflate, br",
           "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
           "Connection": "keep-alive"}


img_re = re.compile('"thumbURL":"(.*?)"')
img_format = re.compile("f=(.*).*?w")


def file_op(img):
    uuid_str = uuid.uuid4().hex
    tmp_file_name = 'E:/pythonproject/pythonProject/pythonproject/Face/images/mark_zuckerberg/%s.jpeg' % uuid_str
    with open(file=tmp_file_name, mode="wb") as file:
        try:
            file.write(img)
        except:
            pass


def xhr_url(url_xhr, start_num=0, page=5):
    end_num = page * 30
    for page_num in range(start_num, end_num, 30):
        resp = requests.get(url=url_xhr + str(page_num), headers=headers)
        if resp.status_code == 200:
            img_url_list = img_re.findall(resp.text)
            for img_url in img_url_list:
                img_rsp = requests.get(url=img_url, headers=headers)
                file_op(img=img_rsp.content)
        else:
            break
    print("Finished")


if __name__ == "__main__":
    org_url = "https://image.baidu.com/search/acjson?tn=resultjson_com&word={text}&pn=".format(
        text=input("Search text:"))
    xhr_url(url_xhr=org_url, start_num=int(input("Start Page:")), page=int(input("Total Page:")))
