""" 端到端验证:chat_json + run_tool_loop 真调 DeepSeek。"""
import asyncio
from llm_client import llm_client


# 1. 结构化抽取
async def demo_extract():
    text = "张三的电话是 138-0013-8000,邮箱是 zhangsan@example.com,他住在北京。"
    messages = [
        {"role": "system", "content": "你是信息抽取助手，只输出 JSON。"},
        {"role": "user", "content": (
            "从下面的文本提取联系人信息，只输出 JSON，格式："
            '{"name": "姓名", "phone": "电话", "email": "邮箱"}\n'
            f"文本：{text}"
        )},
    ]
    result = await llm_client.chat_json(messages)
    print("抽取结果：", result)


# 2. 工具调用循环

def get_weather(city: str) -> str:
    print(f"[工具被调用] get_weather(city={city})")
    table = {"北京": "晴,25°C", "上海": "多云,28°C"}
    return table.get(city, f"暂无 {city} 的天气数据")


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名,如 北京"}},
            "required": ["city"],
        },
    },
}


async def demo_tool_loop():
    messages = [{"role": "user", "content": "北京和上海今天的天气怎么样？"}]
    answer = await llm_client.run_tool_loop(
        messages,
        tools=[WEATHER_TOOL],
        tool_impls={"get_weather": get_weather}
    )
    print("最终回答: ", answer)


async def main():
    await demo_extract()
    print("-----------")
    await demo_tool_loop()


if __name__ == "__main__":
    asyncio.run(main())