

video_list ="/Users/jimmy/Desktop/lilis/moments/moments_test_list.csv"

with open(video_list) as f:
    video_lines=f.readlines()
    
result_list= "/Users/jimmy/Desktop/lilis/moments/output.csv"

with open(result_list) as f2:
    result_lines=f2.readlines()
    
assert(len(video_lines)==len(result_lines))


def change_video_name(video_name):
    if len(video_name)==5:
        return video_name
    if len(video_name)==4:
        video_name="0"+video_name
    if len(video_name)==3:
        video_name="00"+video_name
    if len(video_name)==2:
        video_name="000"+video_name 
    if len(video_name)==1:
        video_name="0000"+video_name
    return video_name
    
f_write1 = open("/Users/jimmy/Desktop/lilis/moments/submit1.txt","a")

print("len(video_lines)")
print(len(video_lines))
for i in range(len(video_lines)):
    print(i)

    video_name = video_lines[i].split("\n")[0] 
    video_name = change_video_name(video_name)
    
  
    new_video_name = video_name+".mp4"+", "+result_lines[i].split("\n")[0]+"\n"
  
    
    f_write1.write(new_video_name)

f_write1.close()
