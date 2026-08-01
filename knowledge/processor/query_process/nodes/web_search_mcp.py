"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:网络搜索节点
        通过MCP StreamableHttp调用阿里云DashScope百炼联网搜索工具
        bailian_web_search,补充知识库外部互联网资料，丰富LLM回答信息
"""
import json
import asyncio
from typing import Dict, Tuple
from agents.mcp import MCPServerStreamableHttp
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState


class WebSearchMcpNode(BaseNode):
    name: str = "web_search_mcp"

    def process(self, state: QueryGraphState) -> Dict:
        # 1.参数校验
        validateed_rewritten_query, validated_item_names = self._validate_input(state)

        # 2.创建MCP客户端
        # 异步函数调用放在事件循环中，变成同步处理
        web_search_docs = asyncio.run(self._web_mcp(validateed_rewritten_query))

        # 5.封装返回结果
        return {
            "web_search_docs": web_search_docs
        }

    def _validate_input(self, state) -> Tuple[str, list]:
        # 1.参数校验
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"Invalid rewritten_query: {rewritten_query}")
            raise StateFieldError(self.name, "rewritten_query", str)

        item_names = state.get("item_names")
        if not item_names or not isinstance(item_names, list):
            self.logger.error(f"Invalid item_names: {item_names}")
            raise StateFieldError(self.name, "item_names", str)
        return rewritten_query, item_names

    async def _web_mcp(self, validateed_rewritten_query):
        # 1.获取MCP客户端
        async with MCPServerStreamableHttp(
                name="网络搜索",
                params={
                    "url": self.config.mcp_dashscope_base_url,  # MCP 服务端点
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},  # 认证头
                    "timeout": 300,  # 请求超时时间（秒）
                    "terminate_on_close": True,  # 关闭时终止连接
                },
                max_retry_attempts=2,  # 最大重试次数
                cache_tools_list=True,  # 缓存工具列表，避免重复请求
        ) as client:
            # 3.调用 call_tool    bailian_web_search
            execute_tool_result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": validateed_rewritten_query, "count": 3}
            )

            if not execute_tool_result.content:
                return []

            text_obj = execute_tool_result.content[0]
            if not hasattr(text_obj, "text") or not text_obj.text.strip():
                return []

            try:
                json_data = json.loads(text_obj.text)
            except json.JSONDecodeError:
                return []

            pages = json_data.get("pages", [])
            if not isinstance(pages, list):
                pages = []

            web_search_docs = []
            for page in pages:
                snippet = page.get("snippet")
                title = page.get("title")
                url = page.get("url")
                web_search_docs.append(
                    {
                        "snippet": snippet,
                        "title": title,
                        "url": url
                    }
                )
        return web_search_docs


if __name__ == '__main__':

    state = {
        "rewritten_query": "今天的小米汽车的股价是多少",
        "item_names": ["RS-12 数字万用表"]
    }

    web_mcp_search = WebSearchMcpNode()
    result = web_mcp_search(state)

    for r in result.get('web_search_docs', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))

"""
meta=None content=[TextContent(type='text', text='{"pages":[{"snippet":"小米汽车(BK0888) 简介:小米汽车 今开:899.56 讨论 若水棋局_朱永红06-15 05:36 $小米集团-W(01810)$预定小米su7的过程,让我看清了这家公司的底色,损失了5000元定金,但帮我排除了以后投资这家公司可能带来的潜在风险。 手机可以堆料,汽车却需要沉淀,不仅仅是技术层面和科技层面的沉淀,更需要安全和可靠性层面的沉淀。 现在我终于想清楚了,小米投资汽车,从战略","hostname":"无","hostlogo":"https://b.bdstatic.com/searchbox/mappconsole/image/20190805/1239163c-77cc-449e-b91f-9bb1d27e43a7.png","title":"小米汽车(BK0888)","url":"https://xueqiu.com/S/BK0888/relevant"},{"snippet":"小米股价过山车:6000亿说没就没,“一夜”回到造车前 知嘹汽车/费德 小米的股价,又坐了一轮刺激的过山车。进入2026年,当人们把目光投向这家话题不断的公司时,发现其股价已经悄悄跌回了35港元以下,市值徘徊在9000亿港元出头。 这个数字,眼熟得很——差不多就是2024年底小米汽车还没发布时的水平。换句话说,过去一年里一度暴涨的那约6000亿港元市值,已经蒸发得干干净净。资本市场给小米讲的那个宏大造车故事,似乎突然被按下了静音键。 这不仅仅是数字游戏,更直接反映在全球车企的江湖排位上。小米市值曾经一度冲到全球车企第三,风光无限,但现在又被比亚迪反超,跌回了第四。 回看小米市值的狂飙,2023年,小米还是个市值2000多亿港元的手机公司。随着2024年小米SU7(参数|询价|图片)的横空出世,资本市场瞬间沸腾,市值最高冲破了1.5万亿港元。一年多涨了四倍,这哪里是估值?这分明是资本圈为它的跨界造车梦提前预支的溢价。 然而,梦做得太美,醒得也就越突然。2025年站上高峰的小米,最先冒出来的是一堆产品和口碑上的挑战。比如,SU7 Ultra那个听起来很厉害的碳纤维机盖,实际功能和宣传的好像不太一样,甚至为此闹上了法庭。接着,尾灯开裂的品控问题、让车主提前结清尾款的销售政策等等问题,接二连三地冒出来,让不少消费者觉得心里不舒服。 真正砸下重锤的是2026年1月,一天之内连着传出两起涉及小米车辆的事件,虽然公司火速回应说数据正常,但这种消息多了,多少让人心存忧虑。 所以,从1.5万亿高点跌掉将近40%,回到9000亿,这既是市场情绪的退潮,也是投资者在重新审视小米。不过,就算跌了这么多,跟造车前的2000多亿市值比,小米现在还是涨了两倍多。这说明,造车这件事,确实从根本上拔高了大家对它的想象空间。股价的起伏是常态,关键还得看它业务的基本盘能不能接得住这份期待。 核心就看两块——手机和汽车。手机板块,小米在全球市场的基本盘还算稳,冲高端的路子也在继续走。汽车这边,故事则更有戏剧性。2025年第二季度,小米汽车交了8万多台,毛利率还提升到了26%以上;到了第三季度,竟然真的实现了单季盈利。这份成绩单,让一些机构又燃起了希望,甚至预测它2026年销量能翻着跟头涨。","hostname":"易车网","hostlogo":"https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=3557303210,2734739739&fm=195&app=88&f=JPEG?w=200&h=200","title":"小米股价过山车:6000亿说没就没,“一夜”回到造车前","url":"https://news.m.yiche.com/hao/wenzhang/106995298/"},{"snippet":"小米汽车(BK0888) 简介:小米汽车","hostname":"雪球","hostlogo":"https://img.alicdn.com/imgextra/i2/O1CN01n9Ac7I1CnITyH2vsW_!!6000000000125-55-tps-32-32.svg","title":"小米汽车(BK0888)","url":"https://www.xueqiu.com/S/BK0888/notices"}],"request_id":"6a7c6b51-a5d1-97a9-b4ca-8ec487cf1152","status":0}', annotations=None, meta=None)] structuredContent=None isError=False
{
  "snippet": "小米汽车(BK0888) 简介:小米汽车 今开:899.56 讨论 若水棋局_朱永红06-15 05:36 $小米集团-W(01810)$预定小米su7的过程,让我看清了这家公司的底色,损失了5000元定金,但帮我排除了以后投资这家公司可能带来的潜在风险。 手机可以堆料,汽车却需要沉淀,不仅仅是技术层面和科技层面的沉淀,更需要安全和可靠性层面的沉淀。 现在我终于想清楚了,小米投资汽车,从战略",
  "title": "小米汽车(BK0888)",
  "url": "https://xueqiu.com/S/BK0888/relevant"
}
{
  "snippet": "小米股价过山车:6000亿说没就没,“一夜”回到造车前 知嘹汽车/费德 小米的股价,又坐了一轮刺激的过山车。进入2026年,当人们把目光投向这家话题不断的公司时,发现其股价已经悄悄跌回了35港元以下,市值徘徊在9000亿港元出头。 这个数字,眼熟得很——差不多就是2024年底小米汽车还没发布时的水平。换句话说,过去一年里一度暴涨的那约6000亿港元市值,已经蒸发得干干净净。资本市场给小米讲的那个宏大造车故事,似乎突然被按下了静音键。 这不仅仅是数字游戏,更直接反映在全球车企的江湖排位上。小米市值曾经一度冲到全球车企第三,风光无限,但现在又被比亚迪反超,跌回了第四。 回看小米市值的狂飙,2023年,小米还是个市值2000多亿港元的手机公司。随着2024年小米SU7(参数|询价|图片)的横空出世,资本市场瞬间沸腾,市值最高冲破了1.5万亿港元。一年多涨了四倍,这哪里是估值?这分明是资本圈为它的跨界造车梦提前预支的溢价。 然而,梦做得太美,醒得也就越突然。2025年站上高峰的小米,最先冒出来的是一堆产品和口碑上的挑战。比如,SU7 Ultra那个听起来很厉害的碳纤维机盖,实际功能和宣传的好像不太一样,甚至为此闹上了法庭。接着,尾灯开裂的品控问题、让车主提前结清尾款的销售政策等等问题,接二连三地冒出来,让不少消费者觉得心里不舒服。 真正砸下重锤的是2026年1月,一天之内连着传出两起涉及小米车辆的事件,虽然公司火速回应说数据正常,但这种消息多了,多少让人心存忧虑。 所以,从1.5万亿高点跌掉将近40%,回到9000亿,这既是市场情绪的退潮,也是投资者在重新审视小米。不过,就算跌了这么多,跟造车前的2000多亿市值比,小米现在还是涨了两倍多。这说明,造车这件事,确实从根本上拔高了大家对它的想象空间。股价的起伏是常态,关键还得看它业务的基本盘能不能接得住这份期待。 核心就看两块——手机和汽车。手机板块,小米在全球市场的基本盘还算稳,冲高端的路子也在继续走。汽车这边,故事则更有戏剧性。2025年第二季度,小米汽车交了8万多台,毛利率还提升到了26%以上;到了第三季度,竟然真的实现了单季盈利。这份成绩单,让一些机构又燃起了希望,甚至预测它2026年销量能翻着跟头涨。",
  "title": "小米股价过山车:6000亿说没就没,“一夜”回到造车前",
  "url": "https://news.m.yiche.com/hao/wenzhang/106995298/"
}
{
  "snippet": "小米汽车(BK0888) 简介:小米汽车",
  "title": "小米汽车(BK0888)",
  "url": "https://www.xueqiu.com/S/BK0888/notices"
}
"""
