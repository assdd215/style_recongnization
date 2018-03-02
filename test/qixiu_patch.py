import urllib2
import urllib
import os
import json


def translateFormData(str):
    str = str.replace("%20"," ")
    str = str.replace("%2F","/")
    return str

def patch_urls():
    base_url = "http://m2-glider-xiu.pps.tv/v1/ugc/recommend_index.json"
    content = [None] * 7
    content[0] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=1&page_size=50&platform=2_22_233&sign=ceefa6716d1021bab04beb3846ca137a51274700&time=1519804834&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[1] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=2&page_size=50&platform=2_22_233&sign=9357068299ebea47b2d691db5aefe5bca4f287c9&time=1519804861&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[2] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=3&page_size=50&platform=2_22_233&sign=dcb008ab3d6ffb1438e06513b58b71e8c87f816f&time=1519804920&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[3] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=4&page_size=50&platform=2_22_233&sign=fc615bd05eb3b2919bfdde6a520f50bf5675a280&time=1519804922&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[4] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=5&page_size=50&platform=2_22_233&sign=50aed502b2d8e84496bbaf8d70e72702009d36f8&time=1519804923&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[5] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=6&page_size=50&platform=2_22_233&sign=89c00345ba23224a08126e786c274105e8ae5844&time=1519804924&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")
    content[6] = translateFormData("app_key=ugc_android&authcookie=24KrYG5cufm23HazR1J4Eyck5dHLSQ9cvSPrm2BDvNlAaDS5XoAffjm1MjsLIsKEE0gfB5a&device_id=41C85F2A683529549AB5B518F3747790&fingerprint=15373216c3c356428d9dbee7b69c7800957b8bbf00b353ebbdfa69487b0cf197a8&netstat=wifi&page=7&page_size=50&platform=2_22_233&sign=f9aca40a942144b9f7e19667aafc2f31b4f884e7&time=1519804925&ua=Mi%20Note%203__android7.1.1%20QixiuApp%20Version%2F3.2.1&version=3.2.1")

    anchors = {}

    for page in range(len(content)):
        req = urllib2.Request(url=base_url, data=content[page])
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        response = urllib2.urlopen(req)
        result = json.loads(response.read())
        print(page)
        upers_msg_json = result['data']["items"]
        count = 0
        for item in upers_msg_json:
            anchors[str(upers_msg_json[item]['user_id'])] = upers_msg_json[item]['live_image']
            count = count + 1
    return anchors

def download_imgs():
    indexs = patch_urls()
    pic_dir = "imgs"
    if os.path.exists(pic_dir) == False:
        os.mkdir(pic_dir)
    for index in indexs:
        urllib.urlretrieve(indexs[index], os.path.join(pic_dir,str(index)) + ".jpg")


# download_imgs()
download_imgs()