import requests
from bs4 import BeautifulSoup
import time
import json
import os

def append_to_json(new_data, filename, direction=r"D:\program_study\study\QA"):
    path = os.path.join(direction, filename)

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    try:
        with open(path, "r+", encoding="utf-8") as f:
            f.seek(0)
            existing_data = json.load(f)

            if isinstance(existing_data, list):
                existing_data.append(new_data)
            elif isinstance(existing_data, dict):
                existing_data.update(new_data)
            else:
                raise TypeError("JSON 根结构需为列表或字典")

            f.seek(0)
            f.truncate()
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

    except json.JSONDecodeError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([new_data], f, ensure_ascii=False, indent=2)


def multi_page(soup):
    try:
        links = soup.find_all("a", class_="flex items-center rounded-lg px-2.5 py-1 hover:bg-gray-50 dark:hover:bg-gray-800")
        for link in links:
            if "Next" in link.text:
                href = link.get("href")
                return "https://hf-mirror.com" + href if href else None
        return None
    except Exception as e:
        print(f"[警告] 解析下一页失败: {e}")
        return None


def extract_each_page(soup):
    try:
        target_divs = soup.find_all("div", class_="SVELTE_HYDRATER contents")
        for div in target_divs:
            if div.has_attr("data-props"):
                data_props = div["data-props"]
                start_pos = data_props.find("rowIdx")
                if start_pos == -1:
                    continue
                pending_part = data_props[start_pos - 2:]
                end_pos = pending_part.find("truncated")
                if end_pos == -1:
                    continue
                pending_part = pending_part[:end_pos - 3]
                fixed_str = "[{}]".format(pending_part.replace("}{", "},{"))
                try:
                    content = json.loads(fixed_str)
                    return content
                except json.JSONDecodeError as e:
                    print(f"[错误] JSON 解析失败: {e}")
                    return None
        print("没有找到 data-props 属性")
        return None
    except Exception as e:
        print(f"[警告] 页面解析失败: {e}")
        return None


if __name__ == '__main__':
    output_filename = input("请输入文件名：") + ".json"
    initial_url = input("请输入初始URL: ")
    limit_input = input("页数限制(输入all爬取所有页): ")
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    current_url = initial_url
    count = 1
    visited_urls = set()

    if limit_input.lower() == "all":
        limit = float('inf')
    else:
        try:
            limit = int(limit_input)
        except ValueError:
            print("无效的页数限制，默认设为1")
            limit = 1

    while current_url and count <= limit:
        if current_url in visited_urls:
            print("检测到重复URL，停止爬取")
            break
        visited_urls.add(current_url)

        print(f"\n第 {count} 页 | 当前URL: {current_url}")

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            result = extract_each_page(soup)
            if result:
                append_to_json(result, output_filename)
            next_url = multi_page(soup)
            current_url = next_url
            count += 1
            time.sleep(5)

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {str(e)}")
            break
        except Exception as e:
            print(f"发生未知错误: {str(e)}")
            break

    print("爬取结束")
