# 作为算法工程师，你不得不会的32个Linux命令


## **1、cd-切换当前目录**

这是一个最基本，也是最常用的命令，它用于切换当前目录，它的参数是要切换到的目录的路径，可以是绝对路径，也可以是相对路径。

```bash
cd /root   # 切换到目录/root
cd ./path          # 切换到当前目录下的path目录中，“.”表示当前目录    
cd ../path         # 切换到上层目录中的path目录中，“..”表示上一层目录
```


## 2、ls-查看文件与目录

这也是一个非常有用的查看文件与目录的命令，它的参数非常多，下面就列出一些常用的参数：

- -l ：列出长数据串，包含文件的属性与权限数据等
- -a ：列出全部的文件，连同隐藏文件（开头为.的文件）一起列出来（常用）
- -d ：仅列出目录本身，而不是列出目录的文件数据
- -h ：将文件容量以较易读的方式（GB，kB等）列出来
- -R ：连同子目录的内容一起列出（递归列出），等于该目录下的所有文件都会显示出来

**注：这些参数也可以组合使用**

```bash
ls -l   # 以长数据串的形式列出当前目录下的数据文件和目录 
ls -al   # 以长数据串的形式列出当前目录下的数据文件和目录及隐藏文件(常用)
ls -lR  # 以长数据串的形式列出当前目录下的所有文件  
ls -aR # 列出当前目录所有文件，包括子目录

ls -al --block-size=m  # 查看文件大小，其中k,m,g表示单位
```

**相关命令：**

如果想展示树形结构，可使用tree命令

```bash
# 使用yum install tree命令先安装tree
tree # 树形展示当前目录下所有子文件和目录，该命令不显示中文

tree -N # 中文展示

tree -N -L 2 # 遍历两级菜单

tree /home --charset=gbk -L 2 # 设定中文编码

# -a显示所有文件，-C文件与目录清单加上颜色，-L 2遍历两级菜单
tree -aC -L 2
```

`-I` 命令允许你使用正则匹配来排除掉你不想看到的文件夹。

```bash
tree -I "node_modules"

# 也可以使用`|`同时排除掉多个文件夹
# 最后一个使用到正则匹配，这样以`test_`开头的文件夹都不会被显示出来。
tree -I "node_modules|cache|test_*"
```


## 3、grep-分析一行内容过滤筛选

分析一行的信息，若当中有我们所需要的信息，就将该行显示出来，该命令通常与管道命令一起使用，用于对一些命令的输出进行筛选加工等，下面就列出一些常用的参数：

- -a ：将binary文件以text文件的方式查找数据
- -c ：计算找到‘查找字符串’的次数
- -i ：忽略大小写的区别，即把大小写视为相同
- -v ：反向选择，即显示出没有‘查找字符串’内容的那一行

```bash
# 把ls -l的输出中包含字母file（不区分大小写）的内容输出  
ls -l | grep -i file  

# 取出文件/etc/passwd中包含root的行，并把找到的关键字加上颜色  
grep --color=auto 'root' /etc/passwd

# 当我们需要过滤多个文件时，也很管用
# 查看以smart开头的目录下面以smart开头的properties配置文件是否包含kafka
grep --color=auto 'kafka' smart*/smart*.properties
```


## 4、cat-查看文本文件的内容

该命令用于查看文本文件的内容，后接要查看的文件名，通常可用管道与more和less一起使用，从而可以一页页地查看数据，下面就列出一些常用的参数：

- **-n** ：由 1 开始对所有输出的行数编号。
- **-b** ：和 -n 相似，只不过对于空白行不编号。

```bash
cat text | less # 查看text文件中的内容,这条命令也可以使用less text来代替 

cat /etc/redhat-release # 查看操作系统版本号
cat /proc/version  # 查看操作系统版本
cat /etc/os-release # 查看操作系统版本号
cat /etc/*release*  # 查看操作系统版本号,这个命令比较好使

# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数
cat /proc/cpuinfo # 查看CPU信息
# 查看物理CPU个数
cat /proc/cpuinfo | grep "physical id"| sort| uniq| wc -l
# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo | grep "cpu cores"| uniq
# 查看逻辑CPU的个数
cat /proc/cpuinfo | grep "processor" | wc -l
# 查看CPU信息（型号）
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

cat /proc/meminfo # 查看内存信息

# 查看所有用户帐号的信息，包括用户名和密码
# passwd文件由许多条记录组成，每条记录占一行，记录了一个用户帐号的所有信息。
# 每条记录由7个字段组成，字段间用冒号“:”隔开，其格式如下：
# username:password:User ID:Group ID:comment:home directory:shell
cat /etc/passwd 

cat /etc/resolv.conf # 查看DNS

# 过滤有error的行，并输出行号
cat -n app.log | grep 'error'

# 清空 /etc/test.txt 文档内容
cat /dev/null > /etc/test.txt
```


## 5、tail/tailf-从尾部查看文本文件的内容

用于从文件尾部查看文件的内容，有一个常用的参数 -f 常用于查阅正在改变的日志文件。tail -f filename 会把 filename 文件里的最尾部的内容显示在屏幕上，并且不断刷新，只要 filename 更新就可以看到最新的文件内容。

常用参数如下：

- -f：循环读取
- -c<数目>：显示的字节数
- -n<行数>：显示文件的尾部 n 行内容

```bash
# 查看文件的后10行
tail -10 /etc/passwd 或 tail -n 10 /etc/passwd 

# 不停地去读/var/log/messages文件最新的内容，这样有实时监视的效果，用Ctrl＋c来终止！
tail -f /var/log/messages 

# 显示文件 notes.log 的内容，从第 20 行至文件末尾:
tail +20 notes.log

# 显示文件 notes.log 的最后 10 个字符:
tail -c 10 notes.log
```

tailf 等同于tail -f -n 10，与tail -f不同的是，如果文件不增长，它不会去访问磁盘文件，所以tailf特别适合那些便携机上跟踪日志文件，因为它减少了磁盘访问，可以省电。


## **6、find-查找文件**

一个查找文件的命令，相对而言，它的使用也相对较为复杂，参数也比较多，所以在这里将给把它们分类列出。

```bash
# 命令格式
find [PATH] [option] [action]  

# 与时间有关的参数：  
-mtime n : n为数字，意思为在n天之前的“一天内”被更改过的文件
-mtime +n : 列出在n天之前（不含n天本身）被更改过的文件名
-mtime -n : 列出在n天之内（含n天本身）被更改过的文件名
-newer file : 列出比file还要新的文件名

# 例如：  
find /root -mtime 0 # 在当前root目录下查找今天之内有改动的文件  
find /root -newer xxx # 在当前root目录下查找比file还要新的文件名


# 与用户或用户组名有关的参数：  
-user name : 列出文件所有者为name的文件  
-group name : 列出文件所属用户组为name的文件  
-uid n : 列出文件所有者为用户ID为n的文件  
-gid n : 列出文件所属用户组为用户组ID为n的文件

# 例如：  
find /home/liguodong -user liguodong # 在目录/home/liguodong 中找出所有者为liguodong的文件  
  
# 与文件权限及名称有关的参数：  
-name filename ：找出文件名为filename的文件  
-size [+-]SIZE ：找出比SIZE还要大（+）或小（-）的文件  
-tpye TYPE ：查找文件的类型为TYPE的文件，TYPE的值主要有：一般文件（f)、设备文件（b、c）、  
             目录（d）、连接文件（l）、socket（s）、FIFO管道文件（p）；  
-perm mode ：查找文件权限刚好等于mode的文件，mode用数字表示，如0755；  
-perm -mode ：查找文件权限必须要全部包括mode权限的文件，mode用数字表示  
-perm +mode ：查找文件权限包含任一mode的权限的文件，mode用数字表示  

# 例如：  
find / -name passwd # 查找文件名为passwd的文件  
find . -perm 0755 # 查找当前目录中文件权限的0755的文件, 第一个0，表示十进制  
find . -size +1k # 查找当前目录中大于1KB的文件，注意c表示byte  
find . -size +1024c # 查找当前目录中大于1KB的文件
```


## 7、locate-查找文件

用于查找符合条件的文档，他会去保存文档和目录名称的数据库内，查找合乎范本样式条件的文档或目录。当我们不知道某个文件放哪里了，能够通过他快速定位到文件目录，这是一条非常有用的命令。

```bash
# 查找eureka-server-xxx.jar文件的位置
locate 'eureka-server-xxx.jar'

# 查询包含passwd的目录或文档
locate passwd
```

**locate与find的不同:**

find 是去硬盘找，locate 只在/var/lib/slocate资料库中找。locate的速度比find快，它并不是真的查找，而是查数据库，一般文件数据库在/var/lib/slocate/slocate.db中，所以locate的查找并不是实时的，而是以数据库的更新为准，一般是系统自己维护，也可以手工升级数据库 ，命令为：

```bash
locate -u
```


## 8、cp-复制文件

该命令用于复制文件，它还可以把多个文件一次性地复制到一个目录下，它的常用参数如下：

- -a ：将文件的特性一起复制
- -p ：连同文件的属性一起复制，而非使用默认方式，与-a相似，常用于备份
- -i ：若目标文件已经存在时，在覆盖时会先询问操作的进行
- -r ：递归持续复制，用于目录的复制行为
- -u ：目标文件与源文件有差异时才会复制

```bash
cp -a file1 file2  # 连同文件的所有特性把文件file1复制成文件file2  
cp file1 file2 file3 dir  #把文件file1、file2、file3复制到目录dir中
```


## 9、scp-远程复制文件

该命令用于 Linux 之间复制文件和目录。 scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。scp 是加密的。

```bash
# 命令格式
scp [可选参数] file_source file_target

scp local_file remote_username@remote_ip:remote_folder 
scp local_file remote_username@remote_ip:remote_file 
scp local_file remote_ip:remote_folder 
scp local_file remote_ip:remote_file
```

下面就列出一些常用的参数：

- -p：保留原文件的修改时间，访问时间和访问权限。
- -q： 不显示传输进度条。
- -r： 递归复制整个目录。
- -v：详细方式显示输出。

```bash
# 从本地服务器复制到远程服务器
# 复制文件
scp /home/space/music/1.mp3 root@www.runoob.com:/home/root/others/music 
# 复制文件夹
scp -r /home/space/music/ root@www.runoob.com:/home/root/others/ 

# 从远程服务器复制到本地服务器
# 复制文件
scp root@www.runoob.com:/home/root/others/music /home/space/music/1.mp3 
# 复制文件夹
scp -r www.runoob.com:/home/root/others/ /home/space/music/
```


## 10、vim-编辑文件

该命令主要用于文本编辑，它接一个文件名作为参数，如果文件存在就打开，如果文件不存在就以该文件名创建一个文件。vim是一个非常好用的文本编辑器，它里面有很多非常好用的命令，在这里不再多说。

```bash
vim demo.txt # 编辑demo.txt文件
```


## 11、mv-移动目录文件

该命令用于移动文件、目录或更名，move之意，它的常用参数如下：

- -f ：force强制的意思，如果目标文件已经存在，不会询问而直接覆盖
- -i ：若目标文件已经存在，就会询问是否覆盖
- -u ：若目标文件已经存在，且比目标文件新，才会更新

**注：** 该命令可以把一个文件或多个文件一次移动一个文件夹中，但是最后一个目标文件一定要是“目录”。

```bash
mv file1 file2 file3 dir # 把文件file1、file2、file3移动到目录dir中  
mv file1 file2  # 把文件file1重命名为file2
mv xxdir/ tempdir # 把目录xxdir下的所有文件和目录移动到temp目录下,包含xxdir目录
mv xxdir/* tempdir # 把目录xxdir下的所有文件和目录移动到temp目录下,不包含xxdir目录
```


## 12、rm-删除目录文件

该命令用于删除文件或目录，remove之意，它的常用参数如下：

- -f ：就是force的意思，忽略不存在的文件，不会出现警告消息
- -i ：互动模式，在删除前会询问用户是否操作
- -r ：递归删除，最常用于目录删除，它是一个非常危险的参数

```
rm -i file # 删除文件file，在删除之前会询问是否进行该操作  
rm -rf dir # 强制删除目录dir中的所有文件
```

**避免rm -rf ***

```
cd ${log_path}
rm -rf *
```

上面的命名，先进入到日志目录，然后把日志都删除。看上去没有任何问题。但是，当目录不存在时，悲剧就发生了。

ps：偷偷告诉你一个小故事，我之前就犯过这种错误，不过好在是自己学习虚拟机上面。如果实现线上生产环境，就不是故事了，而是事故了。

上诉问题有四种方法进行规避：

**第一、命令替换**，在生产环境把rm -rf 命令替换为mv，再写个定时shell定期清理。模拟了回收站的功能。

**第二、收拢权限**，帐号权限的分离，线上分配work帐号，只能够删除/home/work/logs/目录，无法删除根目录。大公司一般线上权限管理比较规范，小公司就未必了，搞不好所有的小伙伴都有权限在线上乱搞。

**第三、使用&&** ，可以通过“&&”，将

```bash
cd ${log_path}
rm -rf *
```

合并成一个语句

```bash
cd ${log_path} && rm -rf *
```

当前半句执行失败的时候，后半句不再执行。

**第四、判断目录是否存在**，制定编码规范，对目录进行操作之前，要先判断目录是否存在。靠人的自觉来保证规范的执行，总感觉有些不太靠谱。当然，规范是有必要的。


## 13、ln-连接

这是一个非常重要命令，它的功能是为某一个文件在另外一个位置建立一个同步的链接。当我们需要在不同的目录，用到相同的文件时，我们不需要在每一个需要的目录下都放一个必须相同的文件，我们只要在某个固定的目录，放上该文件，然后在其它的目录下用ln命令链接它就可以，不必重复的占用磁盘空间。

而链接又可分为两种 : 硬链接(hard link)与软链接(symbolic link)，硬链接的意思是一个档案可以有多个名称，而软链接的方式则是产生一个特殊的档案，该档案的内容是指向另一个档案的位置。硬链接是存在同一个文件系统中，而软链接却可以跨越不同的文件系统。不论是硬链接或软链接都不会将原本的档案复制一份，只会占用非常少量的磁碟空间。

它的常用参数如下：

- -f : 链接时先将与 dist 同档名的档案删除
- -i : 在删除与 dist 同档名的档案时先进行询问
- -n : 在进行软链接时，将 dist 视为一般的档案
- -s : 进行软链接(symbolic link)
- -v : 在连结之前显示其档名
- -b : 将在链接时会被覆写或删除的档案进行备份

```bash
# 创建软链接在/usr/bin/目录下freeswitch文件，
# 如果 /usr/local/freeswitch/bin/freeswitch丢失，/usr/bin/freeswitch将失效
ln -sf /usr/local/freeswitch/bin/freeswitch /usr/bin/


# 删除软链接,和删除普通的文件是一眼的，删除都是使用rm来进行操作
rm -rf 软链接名称

# 下面我们来创建test_chk目录的软链接
ln -s test_chk test_chk_ln

#软链接创建好了，我们来看看怎么删除它
# 正确的删除方式（删除软链接，但不删除实际数据）
rm -rf  ./test_chk_ln
# 错误的删除方式
rm -rf ./test_chk_ln/ (这样就会把原来test_chk下的内容删除)
```

**软链接与硬链接的区别：**

软链接：

- 软链接，以路径的形式存在。类似于Windows操作系统中的快捷方式
- 软链接可以 跨文件系统 ，硬链接不可以
- 软链接可以对一个不存在的文件名进行链接
- 软链接可以对目录进行链接

硬链接：

- 硬链接，以文件副本的形式存在。但不占用实际空间。
- 不允许给目录创建硬链接
- 硬链接只有在同一个文件系统中才能创建


## 14、chmod-修改文件权限

chmod用于管理文件或目录的权限，文件或目录权限的控制分别以读取(r)、写入(w)、执行(x)3种，一般的用法如下：

```bash
chmod [-R] abc 文件或目录
```

参数说明如下：

- -R：进行递归的持续更改，即连同子目录下的所有文件都会更改
- abc : a,b,c各为一个数字，分别表示User、Group、及Other的权限。

该命令有两种用法。一种是包含字母和操作符表达式的文字设定法；

- u 表示该文件的拥有者，g 表示与该文件的拥有者属于同一个群体(group)者，o 表示其他以外的人，a 表示这三者皆是。
-  
   - 表示增加权限、- 表示取消权限、= 表示唯一设定权限。
- r 表示可读取，w 表示可写入，x 表示可执行，X 表示只有当该文件是个子目录或者该文件已经被设定过为可执行。

另一种是包含数字的数字设定法。其中4表示读，2表示写，1表示可执行，因此7表示读写可执行，5表示读可执行

```bash
chmod g+w file # 向file的文件权限中加入用户组可写权限 
# 文件 file1.txt 设为所有人皆可读取 :
chmod ugo+r file1.txt 

chmod 755 file # 把file的文件权限改变为-rxwr-xr-x 

chmod a=rwx file  等价于 chmod 777 file
chmod ug=rwx,o=x file 等价于 chmod 771 file
```


## 15、chgrp-修改文件所属用户组

该命令用于改变文件所属用户组，它的使用非常简单，它的基本用法如下：

```bash
chgrp [-R] dirname/filename
```

参数说明如下：

- -R ：进行递归的持续对所有文件和子目录更改

```bash
# 递归地把dir目录下中的所有文件和子目录下所有文件的用户组修改为users  
chgrp users -R ./dir
```


## 16、chown-修改文件所属用户、用户组

该命令用于改变文件的所有者，与chgrp命令的使用方法相同，只是修改的文件属性不同

```bash
# 将文件 file1.txt 的拥有者设为 runoob，用户组组 runoobgroup :
chown runoob:runoobgroup file1.txt
# 将目前目录下的所有文件与子目录的拥有者皆设为 runoob，群体的使用者 runoobgroup:
chown -R runoob:runoobgroup *
```


## **17、sz/rz-文件上传下载**

这是Linux/Unix同Windows进行ZModem文件传输的命令行工具。windows端需要支持ZModem的telnet/ssh客户端（比如SecureCRT、XShell）

sz：将选定的文件发送到本地机器

rz：运行该命令会弹出一个文件选择窗口，从本地选择文件上传到Linux服务器

```bash
# 安装
yum install lrzsz

# 从本地上传文件到服务器：
rz

# 从服务器下载一个文件到本地： 
sz filename 
# 从服务器下载多个文件到本地： 
sz filename1 filename2
### 下载dir目录下的所有文件，不包含dir下的文件夹： 
# sz dir/*
```


## **18、yum-包安装卸载**

Linux包管理器 ，解决依赖问题，方便快捷

```bash
yum install <package_name> -y  # 安装包
yum search <package_name> # 搜索包名
yum search all  # 查询所有
yum remove <package_name> -y  # 删除包(不建议用，yum可以解决依赖问题，删除会删除所有包依赖)
yum update # 安装所有软件到最新版本
```


## 19、curl-与服务器之间传输数据

curl是一个非常实用的、用来与服务器之间传输数据的工具；支持的协议包括 (DICT, FILE, FTP, FTPS, GOPHER, HTTP, HTTPS, IMAP, IMAPS, LDAP, LDAPS, POP3, POP3S, RTMP, RTSP, SCP, SFTP, SMTP, SMTPS, TELNET and TFTP)，curl设计为无用户交互下完成工作；curl提供了一大堆非常有用的功能，包括代理访问、用户认证、ftp上传下载、HTTP POST、SSL连接、cookie支持、断点续传等。

```bash
# 下载页面
curl -o index.html http://aiezu.com

# 下载并显示进度条
curl -# -o centos6.8.iso http://mirrors.aliyun.com/centos/6.8/isos/x86_64/CentOS.iso

# 继续完成上次终止的未完成的下载
curl -# -o centos6.8.iso -C - http://mirrors.aliyun.com/centos/6.8/isos/x86_64/CentOS.iso

# 访问一个网页
curl https://www.baidu.com


# http请求，带上用户名和密码
curl -u xxx:xxx http://10.250.xxx.xxx:5400/xxx/health

# http post请求
curl http://xxx.xxx.xxx/shorturl -X POST -d '{"originalUrl":"xxx",
"expire":7776000,
"app":"xxx"
}' --header "Content-Type: application/json"
```


## 20、wget-文件下载

一个下载文件的工具，，我们经常要下载一些软件或从远程服务器恢复备份到本地服务器。wget支持HTTP，HTTPS和FTP协议，可以使用HTTP代理。

```bash
# 使用wget下载单个文件
wget http://www.linuxde.net/testfile.zip

# 下载并以不同的文件名保存
wget -O wordpress.zip http://www.linuxde.net/download.aspx?id=1080

# 使用wget断点续传
wget -c http://www.linuxde.net/testfile.zip

# wget限速下载
wget --limit-rate=300k http://www.linuxde.net/testfile.zip
```


## 21、ps-查看进程运行情况

该命令用于将某个时间点的进程运行情况选取下来并输出，它的常用参数如下：

- -a ： 显示除控制进程与无端进程外的所有进程
- -d ：显示除控制进程外的所有进程
- -e ：显示所有进程
- -g ：显示会话或组ID在grplist列表中的进程
- -p ：显示PID在pidlist列表中的进程
- -s ：显示会话ID在sesslist列表中的进程
- -t ：显示终端ID在ttylist列表中的进程
- -u ：显示有效用户ID在userlist列表中的进程
- -x ：以用户为中心组织进程状态信息显示
- -M ：显示进程的安全信息
- -f ：显示完整格式的输出
- -j ：显示任务信息
- -l ：显示长列表
- -o ：仅显示由format指定的列
- -y ：不要显示进程标记
- -L ：显示进程中的线程

其实我们只要记住ps一般使用的命令参数搭配即可，它们并不多，如下：

```bash
ps aux # 查看系统所有的进程数据  
ps ax # 查看不与终端（terminal）有关的所有进程  
ps -lA # 查看系统所有的进程数据  
ps axjf # 查看连同一部分进程树状态 
ps –ef # 显示所有信息，连同命令行 
ps -ef | grep xxx # 过滤池包含xxx的行
```


## 22、top-查看系统的整体运行情况

实时动态地查看系统的整体运行情况，是一个综合了多方信息监测系统性能和运行信息的实用工具。通过top命令所提供的互动式界面，用热键可以管理。它的常用参数如下：

- -b：以批处理模式操作；
- -c：显示完整的命令；
- -i <时间>：设置间隔时间；
- -u <用户名>：指定用户名；
- -p <进程号>：指定进程；
- -n <次数>：循环显示的次数。

top交互命令如下：

- h：显示帮助画面，给出一些简短的命令总结说明；
- k：终止一个进程；
- i：忽略闲置和僵死进程，这是一个开关式命令；
- q：退出程序；
- r：重新安排一个进程的优先级别；
- S：切换到累计模式；
- s：改变两次刷新之间的延迟时间（单位为s），如果有小数，就换算成ms。输入0值则系统将不断刷新，默认值是5s；
- f或者F：从当前显示中添加或者删除项目；
- o或者O：改变显示项目的顺序；
- l：切换显示平均负载和启动时间信息；
- m：切换显示内存信息；
- t：切换显示进程和CPU状态信息；
- c：切换显示命令名称和完整命令行；
- M：根据驻留内存大小进行排序；
- P：根据CPU使用百分比大小进行排序；
- T：根据时间/累计时间进行排序；w：将当前设置写入~/.toprc文件中。

```bash
top  # 显示系统中进程的资源占用状况
top -c # 显示系统中进程的资源占用状况，并显示完整的命令
top -u xxx  # 查看xxx用户的进程的资源占用状况
```

**相关命令：iotop、htop**


## 23、kill-根据进程ID杀死进程

该命令用于向某个工作（%jobnumber）或者是某个PID（数字）传送一个信号，它通常与ps和jobs命令一起使用，它的基本语法如下：

```bash
kill -signal PID
```

signal的常用参数如下：

- 1：SIGHUP，启动被终止的进程
- 2：SIGINT，相当于输入ctrl+c，中断一个程序的进行
- 9：SIGKILL，强制中断一个进程的进行
- 15：SIGTERM，以正常的结束进程方式来终止进程
- 17：SIGSTOP，相当于输入ctrl+z，暂停一个进程的进行

**注：** 最前面的数字为信号的代号，使用时可以用代号代替相应的信号。

```bash
# 以正常的结束进程方式来终于第一个后台工作，可用jobs命令查看后台中的第一个工作进程  
kill -SIGTERM %1   
# 重新改动进程ID为PID的进程，PID可用ps命令通过管道命令加上grep命令进行筛选获得  
kill -SIGHUP PID
# 强制杀死进程号为1112的进程
kill -9 1112
```

**kill -15 PID和kill -9 PID的区别**

kill -9 PID 是操作系统从内核级别强制杀死一个进程。

kill -15 PID 可以理解为操作系统发送一个通知告诉应用主动关闭。效果是正常退出进程，退出前可以被阻塞或回调处理。并且它是Linux缺省的程序中断信号。

尽量使用kill -15 PID而不要使用kill -9 PID。

kill -9 PID没有给进程留下善后的机会：

1. 关闭socket链接
2. 清理临时文件
3. 将自己将要被销毁的消息通知给子进程
4. 重置自己的终止状态

一些磁盘操作多的程序更是不要使用kill -9 PID，会导致数据的丢失，如ES，kafka等。


### 批量杀死进程(ps/grep/awk/kill)

```bash
ps aux|grep server|grep -v grep | awk '{print $2}'|xargs kill -9
```

**说明：**

管道符”|”用来隔开两个命令，管道符左边命令的输出会作为管道符右边命令的输入。

awk的作用是输出某一列，{print $2}就是输出第二列，如上即是pid这一列。

“xargs kill -9” 中的 xargs 命令是用来把前面命令的输出结果作为”kill -9″命令的参数，并执行该命令。”kill -9″会强行杀掉指定进程。


## 24、killall-杀死指定名字的进程

用于杀死指定名字的进程，向一个命令启动的进程发送一个信号，它的一般语法如下：

```bash
killall [-iIe] [command name]
```

它的参数如下：

- -i ：交互式的意思，若需要删除时，会询问用户
- -e ：表示后面接的command name要一致，但command name不能超过15个字符
- -I ：命令名称忽略大小写

```bash
killall -SIGHUP syslogd # 重新启动syslogd
```


## 25、file-辨识文件类型

该命令用于辨识文件类型，因为在Linux下文件的类型并不是以后缀为分的，所以这个命令对我们来说就很有用了，它的用法非常简单，基本语法如下：

```bash
file filename
```

查看test文件格式：

```bash
file ./test
```


## 26、tar-对文件压缩解压缩

该命令用于对文件进行打包，默认情况并不会压缩，如果指定了相应的参数，它还会调用相应的压缩程序（如gzip和bzip等）进行压缩和解压。它的常用参数如下：

- -c ：新建打包文件
- -t ：查看打包文件的内容含有哪些文件名
- -x ：解打包或解压缩的功能，可以搭配-C（大写）指定解压的目录
- -j ：通过bzip2的支持进行压缩/解压缩
- -z ：通过gzip的支持进行压缩/解压缩
- -v ：在压缩/解压缩过程中，将正在处理的文件名显示出来
- -f filename ：filename为要处理的文件
- -C dir ：指定压缩/解压缩的目录dir

**注意：-c,-t,-x不能同时出现在同一条命令中**

通常我们只需要记住下面几条命令即可：

```bash
# 压缩
tar -jcv -f filename.tar.bz2 要被处理的文件或目录名称  
# 查询
tar -jtv -f filename.tar.bz2  
# 解压
tar -jxv -f filename.tar.bz2 -C 欲解压缩的目录  
# 注：上面文件名并不定要以后缀tar.bz2结尾，这里主要是为了说明使用的压缩程序为bzip2

# 解压elasticsearch-5.5.2.tar.gz
tar -zxvf elasticsearch-5.5.2.tar.gz
```


## 27、zip/unzip/gzip/gunzip-对文件压缩解压缩

用于压缩、解压缩文件，zip 压缩的后文件是 *.zip ，而 gzip 压缩后的文件 *.gz，相应的解压缩命令则是 gunzip 和 unzip。

```bash
# 将 /home/html/ 这个目录下所有文件和文件夹打包为当前目录下的 html.zip：
zip -q -r html.zip /home/html
zip -r MiniGPT-4.zip ./MiniGPT-4/

# 如果在我们在 /home/html 目录下，可以执行以下命令：
zip -q -r html.zip *


# 从压缩文件 cp.zip 中删除文件 a.c
zip -dv cp.zip a.c

# 将当前目录下的所有文件和文件夹全部压缩成myfile.zip文件,-r表示递归压缩子目录下所有文件
zip -r myfile.zip ./*

# 把myfile.zip文件解压到 /home/bunny/
# -o:不提示的情况下覆盖文件
# -d:-d /home/bunny 指明将文件解压缩到/home/bunny目录下
unzip -o -d /home/bunny myfile.zip

# 它会将文件压缩为文件 test.txt.gz，原来的文件则没有了，解压缩也一样 
gzip test.txt 

# 它会将文件解压缩为文件 test.txt，原来的文件则没有了，为了保留原有的文件，
# 我们可以加上 -c 选项并利用 linux 的重定向 
gunzip test.txt.gz 

# 这样不但可以将原有的文件保留，而且可以将压缩包放到任何目录中，解压缩也一样 
gzip -c test.txt > /root/test.gz

# 解压缩
gunzip -c /root/test.gz > ./test.txt
```


## 28、adduser/useradd/userdel-增加删除用户

adduser/useradd为创建用户命令，使用权限：系统管理员，root用户。**常用参数说明如下**：

- -c comment：加上备注文字。备注文字会保存在通常是 /etc/passwd）的备注栏位中。
- -d home_dir：设定使用者的根目录为 home_dir ，预设值为预设的 home 后面加上使用者帐号
- -e expire_date：设定此帐号的使用期限（格式为 YYYY-MM-DD），预设值为永久有效
- -f inactive_time：帐号过期几日后永久停权。当值为0时帐号则立刻被停权。而当值为-1时则关闭此功能，预设值为-1
- -g <群组>：指定用户所属的群组。
- -r ：建立一个系统的帐号，这个帐号的 UID 会有限制 (/etc/login.defs)

```bash
# 添加一个一般用户
useradd kk # 添加用户kk

# 为添加的用户指定相应的用户组
useradd -g root kk # 添加用户kk，并指定用户所在的组为root用户组

# 创建一个系统用户
useradd -r kk # 创建一个系统用户kk

# 为新添加的用户指定/home目录
useradd -d /home/myf kk //新添加用户kk，其home目录为/home/myf
# 当用户名kk登录主机时，系统进入的默认目录为/home/myf
```

用户删除命令：userdel，语法如下:

```bash
userdel [login ID]
```

删除用户kk:

```bash
userdel kk
```


## 29、passwd-修改用户密码

更改使用者的密码，常用参数如下：

- -d：删除密码
- -l：停止账号使用
- -S：显示密码信息
- -u：启用已被停止的账户
- -x：设置密码的有效期
- -g：修改群组密码
- -i：过期后停止用户账号

```bash
# 修改用户密码
passwd runoob  # 设置runoob用户的密码

# 显示账号密码信息
passwd -S runoob

# 删除用户密码
passwd -d lx138
```


## **30、time-测算一个命令的执行时间**

该命令用于测算一个命令的执行时间。就像平时输入命令一样，不过在命令的前面加入一个time即可。

在程序或命令运行结束后，在最后输出了三个时间，它们分别是：

- user：用户CPU时间，命令执行完成花费的用户CPU时间，即命令在用户态中执行时间总和；
- system：系统CPU时间，命令执行完成花费的系统CPU时间，即命令在核心态中执行时间总和；
- real：实际时间，从command命令行开始执行到运行终止的消逝时间；

```bash
time ./process.sh # 查看process.sh脚本执行时间
time ps aux # 查看ps aux命令的执行时间
```


## 31、free-显示内存的使用情况

显示内存的使用情况，包括实体内存，虚拟的交换文件内存，共享内存区段，以及系统核心使用的缓冲区等。

**参数说明**：

- -b：以Byte为单位显示内存使用情况。
- -k：以KB为单位显示内存使用情况。
- -m ：以MB为单位显示内存使用情况。
- -g ：以GB为单位显示内存使用情况。
- -o ：不显示缓冲区调节列。
- -s <间隔秒数>：持续观察内存使用状况。
- -t：显示内存总和列。

```bash
# 显示内存使用情况
free # 显示内存使用信息

# 以总和的形式显示内存的使用信息
free -gt # 以总和的形式查询内存的使用信息,以GB为单位

# 周期性的查询内存使用信息
free -g -s 10 # 每10s执行一次命令,以GB为单位
```


## 32、crontab-定时任务

用来定时的去跑一些脚本或者程序，linux内置的cron进程能帮我们实现这些需求，精确到分，设计秒的我们一般自己写脚本。

相关配置文件说明：

- /var/spool/cron/目录下存放的是每个用户包括root的crontab任务，每个任务以创建者的名字命名
- /etc/crontab 这个文件负责调度各种管理和维护任务。
- /etc/cron.d/ 这个目录用来存放任何要执行的crontab文件或脚本。
- 我们还可以把脚本放在/etc/cron.hourly、/etc/cron.daily、/etc/cron.weekly、/etc/cron.monthly目录中，让它每小时/天/星期、月执行一次。

常用参数说明：

- -u ：省略该参数，表示操作当前用户的crontab
- -e：编辑某个用户的crontab文件内容。如果不指定用户，则表示编辑当前用户的crontab文件。
- -l：显示某个用户的crontab文件内容，如果不指定用户，则表示显示当前用户的crontab文件内容。
- -r：从/var/spool/cron目录中删除某个用户的crontab文件，如果不指定用户，则默认删除当前用户的crontab文件。
- -i：在删除用户的crontab文件时给确认提示

**注意：-r，-i尽量不要执行**

常见操作命令如下：

```bash
crontab -e  # 编辑定时任务
* * * * * sh /opt/lampp/test.sh   # 每分钟执行一次test.sh，crontab使用

crontab -l  # 查看定时任务
* * * * * sh /opt/lampp/test.sh

# 重启定时任务进程crond
service crond reload

# 查看日志
# /var/log/cron只会记录是否执行了某些计划的脚本
sudo tail -100f /var/log/cron
```

定时任务配置实例如下：

```bash
# Example of job definition:
# .---------------- minute (0 - 59)
# |  .------------- hour (0 - 23)
# |  |  .---------- day of month (1 - 31)
# |  |  |  .------- month (1 - 12) OR jan,feb,mar,apr ...
# |  |  |  |  .---- day of week (0 - 6) (Sunday=0 or 7) OR sun,mon,tue,wed,thu,fri,sat
# |  |  |  |  |
# *  *  *  *  * user-name  command to be executed

# 每1分钟执行一次myCommand
* * * * * myCommand

# 每小时的第3和第15分钟执行
3,15 * * * * myCommand

# 在上午8点到11点的第3和第15分钟执行
3,15 8-11 * * * myCommand

# 每隔两天的上午8点到11点的第3和第15分钟执行
3,15 8-11 */2  *  * myCommand

# 每周一上午8点到11点的第3和第15分钟执行
3,15 8-11 * * 1 myCommand

# 每晚的21:30重启smb
30 21 * * * /etc/init.d/smb restart

# 每月1、10、22日的4 : 45重启smb
45 4 1,10,22 * * /etc/init.d/smb restart

# 每周六、周日的1 : 10重启smb
10 1 * * 6,0 /etc/init.d/smb restart
# 每天18 : 00至23 : 00之间每隔30分钟重启smb
0,30 18-23 * * * /etc/init.d/smb restart

# 每星期六的晚上11 : 00 pm重启smb
0 23 * * 6 /etc/init.d/smb restart

# 每一小时重启smb
* */1 * * * /etc/init.d/smb restart

# 晚上11点到早上7点之间，每隔一小时重启smb
* 23-7/1 * * * /etc/init.d/smb restart
```
好了，这是2天业余时间整理的日常工作中常用的命令，希望能够对你有帮忙。如果觉得还不错，希望能得到你的一个赞。


# 面试必备20个常用 Linux 命令


# 第一章 什么是linux
> 多用户，多任务，支持多线程和多CPU的操作系统，linux的应用领域：免费，稳定，高效的， 一般运行在大型服务器上

**常用目录介绍**：

| 目录名 | 说明 |
| --- | --- |
| / 根目录 | 一般根目录下只存放目录，有且只有一个根目录 |
| /home 家目录 | 系统默认的家目录，新增用户账号时，用户的家目录都存放在此目录下 |
| /root | 系统管理员root的家目录 |
| /bin/usr/bin | 可执行二进制文件的目录 |
| /etc | 系统配置文件存放的目录 |
| /mnt /media | 光盘默认挂载点 |
| /tmp | 一般用户或正在执行的程序临时存放文件的目录 |
| /var | 这个目录中存放着不断扩充着的东西，我们习惯将那些经常被修改的目录放在这个目录下，包括各种日志文件 |

**[root**[**@localhost **](/localhost )** ~]# ** 的含义：

-  @之前的是当前登录的用户 
-  localhost是主机名字 
-  ~当前所在的位置（所在的目录） 
-  ~家目录 
-  /根目录 
-  #的位置是用户标识 
-  #是超级用户 
-  $普通用户 

`linux的核心思想：一切皆为文件`
**linux命令的写法**：
```
命令名  [选项]  [参数]
```

-  命令名：相应功能的英文单词或单词的缩写 
-  选项：可以用来对命令进行控制，也可以省略，选项不同，命令的结果不同 
-  参数：传给命令的参数，可以是0个，也可以一个或多个 

**linux注意事项**：

-  1.严格区分大小写 
-  2.有的命令有选项和参数，有的有其一，有的都没有 
-  3.选项的格式一般是 -字母 -单词 字母 
-  4.可以加多个选项，多个选项可以合并（例 -a -b 可以合并成-ab） 
-  5.命令 选项 参数 之间一定要有空格 

# 第二章 linux的基础命令

## 1.pwd 命令
> **功能**：显示用户当前所在的目录

**格式**：`pwd`

## 2.ls 命令
> **功能**：对于目录，该命令列出该目录下的所有子目录与文件。对于文件，将列出文件名以及其他信息

**格式**：`ls [选项][目录或文件]`
**常用选项表**：

| 选项 | 说明 |
| --- | --- |
| -a | 查看当前目录下的文件，包括隐藏文件 |
| -l | 长格式显示文件 |
| -lh | 以方便阅读的长格式显示 |


## 3.cd 命令
> **功能**：改变工作目录。将当前工作目录改变到指定的目录下

**格式**：`cd 目录名`
**常用命令**：

| 命令 | 说明 |
| --- | --- |
| `cd ..` | 返回上一级目录 |
| `cd ../..` | 返回上两级目录 |
| cd ~ | 切换到家目录 |
| cd / | 切换到根目录 |
| cd /home/lx/linux1/ | 绝对路径：从家目录出发，直到想要去的目录 |
| cd …/lx/ | 相对路径：从当前目录出发，直到想去的目录 |


## 4.man 命令
> Linux的命令有很多参数，我们不可能全记住，我们可以通过查看联机手册获取帮助。访问Linux手册页的命令是man

格式：`man 其他命令`

## 5.grep 命令
> **功能**：用于查找文件里符合条件的字符串

**格式**：`grep [选项] '查找字符串' 文件名`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -a | 将binary文件以text文件的方式查找数据 |
| -c | 计算找到 ‘查找字符串’ 的次数 |
| -i | 忽略大小写的区别，即把大小写视为相同 |
| -v | 反向选择，即显示出没有 ‘查找字符串’ 内容的那一行 |


## 6.find 命令
> **功能**：用来在指定目录下查找文件

**格式**：`find [路径] [选项] 操作`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -name test | 查询指定目录下,命名为test的文件 |
| -size +100k | 查询指定目录下，文件大于100K的文件 |
| -ctime n | 查询指定目录下，在过去n天内被修改过的文件 |


## 7.chmod 命令
[K’mɒud]
> **功能**：控制用户对文件的权限的命令

**格式**：`chmod [选项] 文件名`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -r | 赋予读取权限 |
| -w | 赋予写入权限 |
| -x | 赋予执行权限 |
| 777 | 赋予可读、可写、可执行权限`（读：4，写：2，执行：1）` |

**权限说明**：（例：`-rw-r--r-x` 的权限为645）

-  权限显示位一共为10位，分为四段，从第二位算起，每三个一组 
-  第1位代表文件类型（`-`表示为普通文件） 
-  第2-4位代表文件所属用户拥有的权限（`rw-`：4+2=6） 
-  第5-7位代表该用户所属组拥有的权限（`-r--`：4） 
-  第8-10位代表其他用户拥有的权限（`r-x`：4+1=5） 

## 8.ps 命令
> **功能**：用来列出系统中当前正在运行的那些进程，类似于 windows 的任务管理器。

**格式**：`ps [选项]`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -A | 列出所有的进程 （重要） |
| -ef | 查看全格式的全部进程 （重要） |
| -w | 显示加宽可以显示较多的资讯 |
| -au | 显示较详细的资讯 |
| -aux | 显示所有包含其他使用者的行程 |


## 9.kill 命令
> 功能：用于删除执行中的程序或工作

格式：`kill [选项]/[信号] 进程号`
常用选项：

| 选项 | 说明 |
| --- | --- |
| -l | 参数会列出全部的信息名称。 |
| -s | 指定要送出的信息。 |

常用信号：

| 信号 | 说明 |
| --- | --- |
| -1 (HUP) | 重新加载进程 |
| -9 (KILL) | 杀死一个进程。(重点) |
| -15 (TERM) | 正常停止一个进程。 |


## 10.tail 命令
> 功能：查看测试项目的日志
说明：一般测试的项目里面，有个logs的目录文件，会存放日志文件，有个xxx.out的文件，可以用tail -f 动态实时查看后端日志

格式：`tail [选项] 文件名`
常用选项：

| 选项 | 说明 |
| --- | --- |
| -f | 实时读取 |
| -1000 | 查看最近1000行日志 |


## 11.netstat 命令
> 功能：查看端口

格式：`netstat \-anp | grep 端口号`

## 12.date 查看当前系统时间
```
 date '+%a' 星期几
       +%A  星期几
       +%b   月份 
       +%B   月份
       +%c   直接显示日期与时间
       +%d   日
       +%D   直接显示日期
       +%F   日期(yyyy-mm-dd)
```

## 13.echo 打印 选项 -e
打印常量 直接打印
打印变量 变量前加$
打印命令 用反引号把命令引起来
终端间传递信息 echo 内容>/dev/pts/终端号
```
echo -e  "要打印的东西  \c"
```

## 14.ping 地址 检测是否与主机连通
格式：`ping 地址`
> 问答题：遇到一个不认识的命令式怎么办

1.man 命令名
2.命令名 – help
3.info cat 命令名 （查看命令的功能，来源，选项等）
4.whatis 命令名
5.通过网络途径

# 第三章 文件操作的命令

## 1.mkdir 命令
> **功能**：创建空目录

**格式**：`mkdir [选项] [路径] 文件名`
**常用选项表**：

| 选项 | 说明 |
| --- | --- |
| -p | 层级创建 |
| -v | 显示创建顺序 |


## 2.rmdir 命令
> **功能**：删除空目录 不能删除非空目录，不能删除文件

**格式**：`rmdir [-p] [路径] 目录名`
常用选项表：

| 选项 | 说明 |
| --- | --- |
| -p | 当子目录被删除后如果父目录也变成空目录的话，就连带父目录一起删除 |


## 3.touch 命令
> **功能**：新建空文件

**格式**：`touch [路径] 文件名 （可以多个）`

## 4.rm 命令
> **功能**：删除文件或目录

**格式**：`rm [选项] 文件名`
**常用选项表**：

| 选项 | 说明 |
| --- | --- |
| -f | 强制删除 |
| -r | 多级删除 |
| -rf | 强制删除给定目录下所有文件和目录 |

**rm 和 rmdir 的区别**：

-  rm 有选项， rmdir 没有选项 
-  rmdir 只能删除空目录，不能删文件 
-  rm 带上选项-r可以删除非空目录 

## 5.mv 命令
> **功能**：mv命令是move的缩写，可以用来移动文件或者将文件改名（move(rename)files），是Linux系统下常用的命令，经常用来备份文件或者目录

**格式**：`mv [选项] [路径] 旧文件名 [新路径][新文件名]`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -f | force 强制的意思，如果目标文件已经存在，不会询问而直接覆盖 |
| -i | 若目标文件 (destination) 已经存在时，就会询问是否覆盖 |

**注意**：

-  如果只移动不改名字，新名字可以不写 
-  如果移动的同时改名字，新名字一定要写 

## 6.cp 命令
> **功能**: 复制文件或目录
**说明**：cp指令用于复制文件或目录，如同时指定两个以上的文件或目录，且最后的目的地是一个已经存在的目录，则它会把前面指定的所有文件或目录复制到此目录中。若同时指定多个文件或目录，而最后的目的地并非一个已存在的目录，则会出现错误信息

**格式**：`cp [选项] [路径] 旧文件名 [新路径][新文件名]`
**常用选项表**：

| 选项 | 说明 |
| --- | --- |
| -f 或 --force | 强行复制文件或目录， 不论目的文件或目录是否已经存在 |
| -i 或 --interactive | 覆盖文件之前先询问用户 |
| -r | 递归处理，将指定目录下的文件与子目录一并处理。若源文件或目录的形态，不属于目录或符号链接，则一律视为普通文件处理 |
| -R 或 --recursive | 递归处理，将指定目录下的文件及子目录一并处理 |


## 7 cat 命令
> **功能**：查看目标文件的内容

**格式**：`cat [选项] 文件名`
**常用选项**：

| 选项 | 说明 |
| --- | --- |
| -b | 对非空输出行编号 |
| -n | 对输出的所有行编号 |
| -s | 不输出多行空行 |


# 第四章 vi/vim

## vi/vim 的使用

> 基本上 vi/vim 共分为三种模式，分别是命令模式（Command mode），输入模式（Insert mode）和底线命令模式（Last
line mode）。


**三种模式的转换图**：
![640](./img/W5yQTOxershmW6di/1722921563185-2e125dac-4d28-4b4d-b1fb-fc8a472518d5-775375.png)

### 命令模式
> 用户刚刚启动 vi/vim，便进入了命令模式。此状态下敲击键盘动作会被Vim识别为命令，而非输入字符。比如我们此时按下i，并不会输入一个字符，i被当作了一个命令。

常用的几个命令：

-  i 切换到输入模式，以输入字符。 
-  x 删除当前光标所在处的字符。 
-  : 切换到底线命令模式，以在最底一行输入命令。 

若想要编辑文本：启动Vim，进入了命令模式，按下i，切换到输入模式。

### 输入模式
> 在命令模式下按下 `i` 就进入了输入模式。

在输入模式中，可以使用以下按键：

-  字符按键以及Shift组合：输入字符 
-  ENTER：回车键，换行 
-  BACK SPACE：退格键，删除光标前一个字符 
-  DEL：删除键，删除光标后一个字符 
-  方向键：在文本中移动光标 
-  HOME/END：移动光标到行首/行尾 
-  Page Up/Page Down：/下翻页 
-  Insert：切换光标为输入/替换模式，光标将变成竖线/下划线 
-  ESC：退出输入模式，切换到命令模式 

### 底线命令模式
> 在命令模式下按下`:`（英文冒号）就进入了底线命令模式。

底线命令模式可以输入单个或多个字符的命令，可用的命令非常多。
在底线命令模式中，基本的命令有（已经省略了冒号）：

-  q 退出程序 
-  w 保存文件 

按ESC键可随时退出底线命令模式




> 原文: <https://www.yuque.com/lucky-bk3s1/sc1v5b/bc8xdksks7wqdtnl>

> 原文: <https://www.yuque.com/lucky-bk3s1/sc1v5b/svimtpzkrbzd89hb>