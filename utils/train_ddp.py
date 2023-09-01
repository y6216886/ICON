import numpy as np
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import time
import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,default="/mnt/cephfs/dataset/NVS/nerfInWild/brandenburg_gate/",
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['blender', 'phototourism'],
                        help='which dataset to train/val')


# def send_mail(subject='No-reply', message='No-reply'):
#     email_host = 'smtp.qq.com'  # 服务器地址
#     sender = '448451652@qq.com'  # 发件人
#     password = 'yyf752952'  # 密码，如果是授权码就填授权码
#     receiver = '448451652@qq.com'  # 收件人

#     msg = MIMEMultipart()
#     msg['Subject'] = subject  # 标题
#     msg['From'] = ''  # 发件人昵称
#     msg['To'] = ''  # 收件人昵称
#     mail_msg = '''<p>\n\t {}</p>'''.format(message)
#     msg.attach(MIMEText(mail_msg, 'html', 'utf-8'))

#     # 发送
#     smtp = smtplib.SMTP()
#     smtp.connect(email_host, 25)
#     smtp.login(sender, password)
#     smtp.sendmail(sender, receiver, msg.as_string())
#     smtp.quit()
#     print('success')
import smtplib
from email.mime.text import MIMEText
from email.header import Header
def send_mail(num_gpu,memorylist):
  # -*- coding: utf-8 -*-

    
    mail_host="smtp.qq.com"#设置的邮件服务器host必须是发送邮箱的服务器，与接收邮箱无关。
    mail_user="476391708@qq.com"#qq邮箱登陆名
    mail_pass="czigyfgielnzbifj" #开启stmp服务的时候并设置的授权码，注意！不是QQ密码。
    
    sender='448451652@qq.com'#发送方qq邮箱
    receivers=['448451652@qq.com']#接收方qq邮箱
    
    # message=MIMEText(message,'plain','utf-8')
    

    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText('{} gpu025 gpu(s) free {} memory'.format(num_gpu,memorylist), 'plain', 'utf-8')
    message['From'] = Header("Yif", 'utf-8')   # 发送者
    message['To'] = Header("Yif", 'utf-8')        # 接收者

    subject = '{} gpu025 gpu(s) free {} memory'.format(num_gpu,memorylist)
    message['Subject'] = Header(subject, 'utf-8')


    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect("smtp.qq.com")   # 连接 qq 邮箱
        smtpObj.login(sender, mail_pass)   # 发送者账号和授权码
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException:
        print("Error: 无法发送邮件")


def get_gpu_memory():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
    memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    os.system('rm tmp.txt')
    return memory_gpu


if __name__=="__main__":
  hparams = get_opts()
  flag_last = get_gpu_memory()
  while True:
      gpu_memory = get_gpu_memory()
      print("gpu free memory:{} ".format(gpu_memory))

      flag =  np.array(gpu_memory)
      num_changed = np.linalg.norm(np.sign(flag - flag_last), ord=1)
      # 如果有一块卡显存改变
      if num_changed> 0 and max(gpu_memory)>12000:
          while True:
              try:
                  send_mail(num_changed,gpu_memory)
                  # send_mail("{} gpu(s) has changed".format(num_changed), "gpu free memory: {} ".format(gpu_memory))
                  break
              except:
                  print('warning: email not sent.')
          time.sleep(7200)

      flag_last = flag
      time.sleep(240)
