"""
  @Author:lining-lo
  @Time:2026/7/24
  @Desc:答案生成节点
        区分两种场景：使用上游已有答案 / 调用LLM生成新答案。
        兼容流式SSE推送与非流式同步返回；自动拼装上下文、控制文本长度限制；
        对话结束后将用户提问、回答持久化保存至会话历史库。
"""
from typing import Dict, List, Any, Tuple
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        答案生成节点：
            1.有答案
                1.1 流式
                    push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta_text})
                    push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
                1.2 非流式
                    set_task_result(state['task_id'], "answer", state['answer'])
            2.无答案
                2.1 流式
                    push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta_text})
                    push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
                2.2 非流式
                    set_task_result(state['task_id'], "answer", state['answer'])
        :param state:
            session_id,
            task_id,
            is_stream,
            rewritten_query,
            item_names,
            reranked_docs,
            history
        :return:
            {
                "answer": ""
            }
        """
        is_stream = state.get("is_stream")
        answer = state.get("answer")
        # 1.已有答案处理
        if answer:
            # 商品名称确认节点，生成了答案： 中置信选择或抱歉
            # state["answer"] = f"我不确定您指的是哪款产品。您是在询问以下产品吗：{'、'.join(options)}？"
            # state["answer"] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            self._push_existing_answer(state)
        else:
            # 2.没有答案
            # 2.1 准备提示词模板
            prompt = self._generate_prompt(state)
            state["prompt"] = prompt

            # 2.2 调用LLM大模型生成答案
            self._call_llm(state)

        # 3.保存历史记录
        self._write_history(state)

        # 4.发送最终流式终止事件
        if is_stream:
            push_sse_event(task_id=state['task_id'], event=SSEEvent.FINAL, data={"answer": state['answer']})

        return state

    def _push_existing_answer(self, state):
        if not state.get("is_stream"):
            set_task_result(task_id=state['task_id'], key="answer", value=state['answer'])
        # else: # 这里省略代码。 由纪录历史之后，统一统一发送final流式事件。
        #     push_sse_event(task_id=state['task_id'], event=SSEEvent.FINAL, data={"answer":state['answer']})

    def _generate_prompt(self, state) -> str:
        """
        创建提示词模板
        :param state:
        :return:
        """
        budget_chars = self.config.max_context_chars  # 12000

        # 拼串后上下文，可继续拼接剩余字符长度。
        context_str, budget = self._format_reranked_docs(state.get("reranked_docs", []), budget_chars)

        # 最近10条历史会话。5轮对话。
        history_str, budget = self._format_chat_history(state.get("history", []), budget)

        return ANSWER_PROMPT.format(
            context=context_str,
            history=history_str,
            item_names=state.get("item_names", ""),
            question=state.get("rewritten_query", "") or state.get("original_query", "")
        )

    def _call_llm(self, state):
        """
        根据提示词模板调用LLM生成答案
        :param state:
        :return: 流式或非流式
        """

        try:
            llm_client = AIClients.get_llm_openai(False)
        except Exception as e:
            self.logger.warning(f"获取LLM客户端失败:{e}")
            raise ConnectionError("获取LLM客户端失败")

        prompt = state.get("prompt", "")
        if state.get("is_stream"):
            state["answer"] = self._call_llm_stream(state, llm_client, prompt)
        else:
            state["answer"] = self._call_llm_invoke(llm_client, prompt)
            set_task_result(task_id=state.get("task_id"), key="answer", value=state["answer"])

    def _format_reranked_docs(self, reranked_docs: List[Dict[str, Any]], char_budget: int) -> Tuple[str, int]:
        """
        拼串：
            [1][source="local"][chunk_id=str(467775438085177876)][title="万用表RS-12的使用"][score=0.99999] \n xxxxxxxxxxxxxxx内容xxxxxxxxxxxxxxxxxxx \n\n
            [2][source="web"][url="http://xxxx"][title="万用表RS-12的使用"][score=0.99999] \n xxxxxxxxxxxxxxx内容xxxxxxxxxxxxxxxxxxx \n\n
        :param param:
        :param budget_chars:
        :return:
        """
        """格式化重排序文档，带字符预算控制"""
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"),
                ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"),
                ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    def _format_chat_history(self, chat_history: List[Dict[str, Any]], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话"""
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"

            if used_chars + len(formatted_line) > char_budget:
                break

            formatted_lines.append(formatted_line)
            used_chars += len(formatted_line) + 1

        return "\n".join(formatted_lines), char_budget - used_chars

    def _call_llm_stream(self, state, llm_client, prompt) -> str:
        try:
            accumulate_answer = ""  # 积累答案。
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulate_answer += delta_text
                    push_sse_event(task_id=state.get("task_id"), event=SSEEvent.DELTA, data={"delta": delta_text})
            return accumulate_answer
        except Exception as e:
            self.logger.warning(f"LLM-流式生成答案失败了:{e}")
            return "LLM-流式生成答案失败了"

    def _call_llm_invoke(self, llm_client, prompt) -> str:
        try:
            llm_response = llm_client.invoke(prompt)
            return llm_response.content.strip()
        except Exception as e:
            self.logger.warning(f"LLM-非流式生成答案失败了:{e}")
            return "LLM-非流式生成答案失败了"

    def _write_history(self, state):
        """
        问：什么时候记录历史会话？什么时候使用到了历史会话？
            使用历史记录：
                商品名确认节点。拉取最近10条
                生成提示词模板，参数需要历史会话
            生成答案时需要记录历史

        :param state:
        :return:
        """
        # 保存用户问题
        save_chat_message(
            session_id=state.get("session_id"),  # 用户打开聊天窗口,后端生成的。
            role="user",
            text=state.get("rewritten_query") or state.get("original_query"),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names"),
        )

        # 保存助手答案
        save_chat_message(
            session_id=state.get("session_id"),  # 用户打开聊天窗口,后端生成的。
            role="assistant",
            text=state.get("answer", ""),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names"),
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import json

    load_dotenv()

    from knowledge.processor.query_process.base import setup_logging

    setup_logging()

    print("=" * 60)
    print("开始测试: 答案生成节点 (AnswerOutputNode)")
    print("=" * 60)

    # 构造模拟状态
    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": False,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")
    print("-" * 60)

    # 执行答案生成
    node = AnswerOutputNode()
    result = node.process(mock_state)

    # 打印结果
    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)

    print("\n测试完成")
