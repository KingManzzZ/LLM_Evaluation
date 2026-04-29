import os
import json


file= r"dataset/performance\长文本理解能力\信息提取\information_extraction.json"
with open(file, "r", encoding="utf-8") as f:
   data=json.load(f)
result=[]
count=0
for item in data:

        # trans_item={
        #     "rowIdx":count,
        #     "question": item["context"]+"\n"+item["input"],
        # #     "options":[
        # #         "A:"+item["option1"],
        # #         "B:"+item["option2"]
        # # ],
        #     "answer":i ,
        #     "type_index":"short_answer",
        # }
        #count+=1
        #result.append(trans_item)
        item["question"]=item["question"].replace("answer","final answer")

direction=r"D:\program_study\study\QA\information_extraction.json"
with open(direction, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print("finish")
