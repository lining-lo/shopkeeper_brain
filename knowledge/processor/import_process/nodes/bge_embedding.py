"""
  @Author:lining-lo
  @Time:2026/7/14
  @Desc:切片混合向量化节点
        分批读取带商品名的切片，批量生成稠密+稀疏向量回填切片；
        批量推理提速，异常自动填充空向量兜底，输出带双向量的切片供Milvus入库
"""
from typing import List, Tuple, Dict, Any
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState, create_default_state
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    name: str = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "参数校验")
        # 校验数据
        chunks, embedding_batch_size = self._validate_get_inputs(state)
        self.logger.info(f"待向量化切片总数：{len(chunks)}，批次大小：{embedding_batch_size}")

        # 分批次向量化
        chunks_len = len(chunks)
        all_processed_chunks = []
        for i in range(0, chunks_len, embedding_batch_size):
            batch_chunks = chunks[i:i + embedding_batch_size]
            self.log_step("step2", f"处理批次：{i + 1} ~ {min(i + embedding_batch_size, chunks_len)} / {chunks_len}")
            # 执行向量化
            handled_batch = self._process_batch_chunks(batch_chunks, i, chunks_len)
            all_processed_chunks.extend(handled_batch)

        state["chunks"] = all_processed_chunks
        return state

    def _validate_get_inputs(self, state) -> Tuple[List[Dict[str, Any]], int]:
        chunks = state.get("chunks", None)
        # 配置带默认兜底
        embedding_batch_size = getattr(self.config, "embedding_batch_size", 8)

        # 修正校验逻辑：为空 OR 不是列表则报错
        if not chunks or not isinstance(chunks, list):
            self.logger.error(f"chunks不存在或格式非法，必须为非空列表")
            raise StateFieldError(self.name, "chunks", expected_type=list)

        if not embedding_batch_size or embedding_batch_size <= 0:
            self.logger.error(f"embedding_batch_size不存在或小于等于零")
            raise ValidationError(f"embedding_batch_size不存在或小于等于零", self.name)

        return chunks, embedding_batch_size

    def _process_batch_chunks(self, batch_chunks: List[Dict[str, Any]], start_idx: int, total_len: int) -> List[
        Dict[str, Any]]:
        # 批量收集文本，一次性推理，充分利用GPU并行
        text_list = []
        for chunk in batch_chunks:
            item_name = chunk.get("item_name", "")
            content = chunk.get("content", "")
            text = f"{item_name}\n{content}".strip()
            text_list.append(text)

        # 批量编码
        dense_list, sparse_dict_list = self._batch_embedding(text_list)

        # 向量回填对应切片
        handled_batch = []
        for idx, chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = dense_list[idx]
            chunk["sparse_vector"] = sparse_dict_list[idx]
            handled_batch.append(chunk)
        return handled_batch

    def _batch_embedding(self, text_list: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """整批文本统一向量化，返回稠密列表、稀疏字典列表"""
        # 初始化兜底空向量
        default_dense = [0.0] * 1024
        default_sparse = {}
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE-M3模型客户端初始化失败: {str(e)}")
            return ([default_dense for _ in text_list], [default_sparse for _ in text_list])

        try:
            embeddings = bge_m3_client.encode_documents(text_list)
            dense_all = embeddings["dense"]
            sparse_csr = embeddings["sparse"]
            sparse_result = []

            # 批量解析CSR矩阵
            indptr = sparse_csr.indptr
            indices = sparse_csr.indices
            data = sparse_csr.data
            for i in range(len(text_list)):
                s = indptr[i]
                e = indptr[i + 1]
                token_ids = indices[s:e].tolist()
                weights = data[s:e].tolist()
                sparse_result.append(dict(zip(token_ids, weights)))

            dense_list = [vec.tolist() for vec in dense_all]
            return dense_list, sparse_result

        except Exception as e:
            self.logger.error(f"BGE-M3批量向量化失败: {str(e)}")
            return ([default_dense for _ in text_list], [default_sparse for _ in text_list])

if __name__ == '__main__':
    setup_logging()
    init = {
        'file_dir': 'D:\\资料',
        'file_title': '查重_简洁报告单',
        'import_file_path': 'D:\\查重_简洁报告单.pdf',
        'is_md_read_enabled': False,
        'is_pdf_read_enabled': True,
        'md_content': '![RS PRO '
                      '品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n'
                      '\n'
                      '使用说明书\n'
                      '\n'
                      'RS-12\n'
                      '\n'
                      '编号: 123-1939\n'
                      '\n'
                      '数字万用表\n'
                      '\n'
                      '![中文操作界面标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/81735c16d6175e1dd624407b3448d8bf2039d8e22b1f9cc3941e533706a070a9.jpg)\n'
                      '\n'
                      'CE\n'
                      '\n'
                      '![RS-12数字万用表面板结构与功能标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/f179d9399297a15b5d4e764602734c25302eec0b528b231f0e455ca9c76dce0b.jpg)\n'
                      '\n'
                      '## 安全手册\n'
                      '\n'
                      '为了您的安全，请在使用本仪表之前仔细阅读该手册:\n'
                      '\n'
                      '使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 '
                      'colspan=1>输入量程</td></tr><tr><td rowspan=1 colspan=1>功能</td><td '
                      'rowspan=1 colspan=1>最大输入</td></tr><tr><td rowspan=1 '
                      'colspan=1>交/直流电压</td><td rowspan=1 '
                      'colspan=1>直流/交流电压600V</td></tr><tr><td rowspan=1 '
                      'colspan=1>直流/交流电压</td><td rowspan=1 colspan=1>直流/交流电压600V, '
                      '200Vrms 用于200mV量程</td></tr><tr><td rowspan=1 '
                      'colspan=1>mA直流</td><td rowspan=1 colspan=1>200mA '
                      '250V快速熔断保险丝</td></tr><tr><td rowspan=1 colspan=1>A DC</td><td '
                      'rowspan=1 colspan=1>10A 250V '
                      '快速熔断保险丝(最多每15分钟，需时30秒)</td></tr><tr><td rowspan=1 '
                      'colspan=1>电阻,短路测试</td><td rowspan=1 colspan=1>250Vrms, '
                      '最多15秒</td></tr></table>\n'
                      '\n'
                      '2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n'
                      '\n'
                      '3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n'
                      '\n'
                      '4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n'
                      '\n'
                      '5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n'
                      '\n'
                      '6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n'
                      '\n'
                      '7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n'
                      '\n'
                      '## 安全标识\n'
                      '\n'
                      '![通用安全警告标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6ed1c2d2c192fe7422f77d0eb13133a4f4b01ea3e738ce2017bc5df75185e1a0.jpg)\n'
                      '\n'
                      '表明此操作须参照说明书进行。\n'
                      '\n'
                      'WARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n'
                      '\n'
                      'CAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n'
                      '\n'
                      '![最大值标识（MAX）警示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/4033d9da04e1ffceb8382efdf1c281a8fbddf7ebe238b3aa131ff2f0c43fbeb0.jpg)\n'
                      '\n'
                      '请勿连接到500VAC或VDC的电路上。\n'
                      '\n'
                      '![闪电警示符号：表示危险电压，需避免接触以防致命伤害或设备损坏](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/2219ca75130874ec766983013be86c7afd233a4a1b3a5188990261156f1f2cc5.jpg)\n'
                      '\n'
                      '表明此端口可能出现危险电压。\n'
                      '\n'
                      '![双绝缘保护标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6b7cc68242e90cec872c128c48320467c4ea101b9973420a5a2a2c46c1a7d489.jpg)\n'
                      '\n'
                      '双绝缘保护。\n'
                      '\n'
                      '## 控制与端口\n'
                      '\n'
                      '1.LCD液晶显示\n'
                      '\n'
                      '2.功能选择转盘\n'
                      '\n'
                      '3.10A端口\n'
                      '\n'
                      '4.COM端口\n'
                      '\n'
                      '5.正极端口\n'
                      '\n'
                      '6.数据保持按键\n'
                      '\n'
                      '7.背光按键\n'
                      '\n'
                      '![万用表各部件标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg)\n'
                      '\n'
                      '## 功能符号指示\n'
                      '\n'
                      '•))) 蜂鸣指示\n'
                      '\n'
                      '![二极管测试指示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/e92ffd955b1ca1fd14290da681a763771c958cbcf0a73a332107f471f96c29b2.jpg)\n'
                      '\n'
                      '二极管测试指示\n'
                      '\n'
                      'µ micro (电流范围)\n'
                      '\n'
                      'm milli ( 电压/电流范围)\n'
                      '\n'
                      'k kilo (电阻范围)\n'
                      '\n'
                      'VDC 直流电压\n'
                      '\n'
                      'VAC 交流电流\n'
                      '\n'
                      'ADC 直流电流\n'
                      '\n'
                      'BAT 电池电量不足指示\n'
                      '\n'
                      '## 规格\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1>功能</td><td rowspan=1 '
                      'colspan=1>量程</td><td rowspan=1 colspan=1>分辨率</td><td rowspan=1 '
                      'colspan=1>精确度</td></tr><tr><td rowspan=5 '
                      'colspan=1>直流电压</td><td rowspan=1 colspan=1>200mV</td><td '
                      'rowspan=1 colspan=1>0.1mV</td><td rowspan=3 colspan=1>± (0.5% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000mV</td><td rowspan=1 '
                      'colspan=1>1mV</td></tr><tr><td rowspan=1 colspan=1>20V</td><td '
                      'rowspan=1 colspan=1>0.01V</td></tr><tr><td rowspan=1 '
                      'colspan=1>200V</td><td rowspan=1 colspan=1>0.1V</td><td '
                      'rowspan=2 colspan=1>± (0.8% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>600V</td><td '
                      'rowspan=1 colspan=1>1V</td></tr><tr><td rowspan=2 '
                      'colspan=1>交流电压</td><td rowspan=1 colspan=1>200V</td><td '
                      'rowspan=1 colspan=1>0.1V</td><td rowspan=2 colspan=1>± (1.2% '
                      'reading + 10 digits50/60Hz)</td></tr><tr><td rowspan=1 '
                      'colspan=1>600V</td><td rowspan=1 colspan=1>1V</td></tr><tr><td '
                      'rowspan=4 colspan=1>直流电流</td><td rowspan=1 '
                      'colspan=1>2000μA</td><td rowspan=1 colspan=1>1μA</td><td '
                      'rowspan=2 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>20mA</td><td '
                      'rowspan=1 colspan=1>10μA</td></tr><tr><td rowspan=1 '
                      'colspan=1>200mA</td><td rowspan=1 colspan=1>100μA</td><td '
                      'rowspan=1 colspan=1>± (1.2% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>10A</td><td '
                      'rowspan=1 colspan=1>10mA</td><td rowspan=1 colspan=1>± (2.0% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=5 '
                      'colspan=1>电阻</td><td rowspan=1 colspan=1>200Ω</td><td '
                      'rowspan=1 colspan=1>0.1Ω</td><td rowspan=4 colspan=1>± (0.8% '
                      'reading + 2 digits)</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000Ω</td><td rowspan=1 '
                      'colspan=1>1Ω</td></tr><tr><td rowspan=1 colspan=1>20kΩ</td><td '
                      'rowspan=1 colspan=1>0.01kΩ</td></tr><tr><td rowspan=1 '
                      'colspan=1>200kΩ</td><td rowspan=1 '
                      'colspan=1>0.1kΩ</td></tr><tr><td rowspan=1 '
                      'colspan=1>2000kΩ</td><td rowspan=1 colspan=1>1kΩ</td><td '
                      'rowspan=1 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=2 colspan=1>电池</td><td '
                      'rowspan=1 colspan=1>9V</td><td rowspan=1 '
                      'colspan=1>10mV</td><td rowspan=2 colspan=1>± (1.0% reading + 2 '
                      'digits)</td></tr><tr><td rowspan=1 colspan=1>1.5V</td><td '
                      'rowspan=1 colspan=1>1mV</td></tr></table>\n'
                      '\n'
                      '注意: 精确度规格由两种因素组成。  \n'
                      '● (% reading) –测量电路的精确度。  \n'
                      '● (+ digits) –数位转换器条码的精确度。  \n'
                      '注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。\n'
                      '\n'
                      '## 技术指标说明\n'
                      '\n'
                      '二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n'
                      '\n'
                      '短路蜂鸣测试 若电阻小于30时产生蜂鸣\n'
                      '\n'
                      '电池测试电流 9V (6mA)；1.5V (100mA)\n'
                      '\n'
                      '输入阻抗 >1MΩ\n'
                      '\n'
                      '交流电压频宽 45Hz～450Hz\n'
                      '\n'
                      'DCA电压跌路测试 200mV\n'
                      '\n'
                      '显示 3 ½ 数位，2000位液晶显示，1.1”数位\n'
                      '\n'
                      '超量程提示 以“1”表示\n'
                      '\n'
                      '极性 自动(正极无显示);负极显示(-)\n'
                      '\n'
                      '测量率 正常情况下每秒2次\n'
                      '\n'
                      '低电池提示 电池电压不足时，显示BAT符号\n'
                      '\n'
                      '电池 一粒9V (NEDA 1604) 电池\n'
                      '\n'
                      '保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n'
                      '\n'
                      '操作环境 32°F～122°F (0°C～50°C)\n'
                      '\n'
                      '储存温度 -4°F～140°F (-20°C～60°C)\n'
                      '\n'
                      '相对湿度 <70% 操作, <80% 储存\n'
                      '\n'
                      '室内使用,最高海拔 7000英尺(2000米)\n'
                      '\n'
                      '重量 255g\n'
                      '\n'
                      '尺寸 150mm x 70mm x 48mm\n'
                      '\n'
                      '安全认证 室内使用，符合过电压类别II\n'
                      '\n'
                      '污染级别 2\n'
                      '\n'
                      '## 电池安装\n'
                      '\n'
                      '警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n'
                      '\n'
                      '1. 把表笔与仪表断开。\n'
                      '\n'
                      '2. 用螺丝刀拧开电池后盖上的螺母。\n'
                      '\n'
                      '3. 正确安装电池，正负极应一致。\n'
                      '\n'
                      '4. 盖上电池后盖并拧紧螺丝钉。\n'
                      '\n'
                      '警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n'
                      '\n'
                      '注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n'
                      '\n'
                      '## 操作指导\n'
                      '\n'
                      '## 数值保持按键HOLD\n'
                      '\n'
                      '保持键允许仪表固定测量值以供参考：\n'
                      '\n'
                      '1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n'
                      '\n'
                      '2. 再次按下“DATA HOLD”键 切换至正常操作\n'
                      '\n'
                      '## 背光灯键（BACKLIGHT）\n'
                      '\n'
                      '1. 按下背光灯键开启背光灯。\n'
                      '\n'
                      '2. 再次按背光灯键关闭背光灯。\n'
                      '\n'
                      '警告：小心触电，高压电流十分危险，应小心操作。\n'
                      '\n'
                      '1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n'
                      '\n'
                      '2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n'
                      '\n'
                      '注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n'
                      '\n'
                      '## 测量非接触交流电压\n'
                      '\n'
                      '警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n'
                      '\n'
                      '1. 让其探头靠近或插入火线的输出插座孔时。\n'
                      '\n'
                      '2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n'
                      '\n'
                      '注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n'
                      '\n'
                      '注意: '
                      '此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n'
                      '\n'
                      '## 直流电压测量\n'
                      '\n'
                      '注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n'
                      '\n'
                      '1. 将功能转盘置于V DC的位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n'
                      '\n'
                      '4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n'
                      '\n'
                      '## 交流电压测量\n'
                      '\n'
                      '警告：谨防触电。\n'
                      '\n'
                      '若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n'
                      '\n'
                      '注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n'
                      '\n'
                      '![交流电压测量操作示意图（表笔连接与读数显示）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n'
                      '\n'
                      '1. 将功能转盘置于V AC的位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '3. 将表笔尖端接触被测物。\n'
                      '\n'
                      '4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n'
                      '\n'
                      '在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n'
                      '\n'
                      '## 直流电流测量\n'
                      '\n'
                      '注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n'
                      '\n'
                      '![直流电流测量接线示意图（10A档位）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM端口。\n'
                      '\n'
                      '2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n'
                      '\n'
                      '3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n'
                      '\n'
                      '4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n'
                      '\n'
                      '5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n'
                      '\n'
                      '6. 接通电源。\n'
                      '\n'
                      '7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA '
                      'DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n'
                      '\n'
                      '![直流10A电流测量接线示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/950918b0a12b239de83d45093fa5e6258bfa3d33848f927ecf0993f3210bf3d9.jpg)\n'
                      '\n'
                      '## 电阻测量\n'
                      '\n'
                      '警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n'
                      '\n'
                      '1. 将功能转盘置于最高电阻Ω位置.\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n'
                      '\n'
                      '3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n'
                      '\n'
                      '4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n'
                      '\n'
                      '![数字万用表电阻测量操作示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)\n'
                      '\n'
                      '## 短路蜂鸣测试\n'
                      '\n'
                      '警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n'
                      '\n'
                      '1. 将功能键转盘置于 位置。\n'
                      '\n'
                      '2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n'
                      '\n'
                      '3. 把表笔与被测物体相接触。\n'
                      '\n'
                      '4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n'
                      '\n'
                      '## 二极管测试\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n'
                      '\n'
                      '2. 将功能转盘置于 位置。\n'
                      '\n'
                      '3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 '
                      '0V，开路时会在两种极性上显示“1”符号。\n'
                      '\n'
                      '## 电池测试\n'
                      '\n'
                      '1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n'
                      '\n'
                      '2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n'
                      '\n'
                      '3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n'
                      '\n'
                      '4. 在显示屏上读取数值。\n'
                      '\n'
                      '<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 '
                      'colspan=1>良好</td><td rowspan=1 colspan=1>较弱</td><td rowspan=1 '
                      'colspan=1>坏的</td></tr><tr><td rowspan=1 colspan=1>9V '
                      '电池：</td><td rowspan=1 colspan=1>&gt;8.2V</td><td rowspan=1 '
                      'colspan=1>7.2 至 8.2V</td><td rowspan=1 '
                      'colspan=1>&lt;7.2V</td></tr><tr><td rowspan=1 colspan=1>1.5V '
                      '电池：</td><td rowspan=1 colspan=1>&gt;1.35V</td><td rowspan=1 '
                      'colspan=1>1.22 至 1.35V</td><td rowspan=1 '
                      'colspan=1>&lt;1.22V</td></tr></table>\n'
                      '\n'
                      '## 更换电池\n'
                      '\n'
                      '警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n'
                      '\n'
                      '1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n'
                      '\n'
                      '2. 按下面的步骤安装电池。\n'
                      '\n'
                      '3. 妥善处理废电池。\n'
                      '\n'
                      '警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n'
                      '\n'
                      '## 更换保险丝\n'
                      '\n'
                      '警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n'
                      '\n'
                      '1. 把表笔与仪表及其它被测物断开。\n'
                      '\n'
                      '2. 用螺丝刀拧开保险丝门上的螺母。\n'
                      '\n'
                      '3. 轻轻取出废旧的保险丝。\n'
                      '\n'
                      '4. 装入新的保险丝。\n'
                      '\n'
                      '5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V '
                      '快速熔断保险丝用于10A的量程。\n'
                      '\n'
                      '6. 盖回后盖，拧紧螺钉。\n'
                      '\n'
                      '警告: 为防触电，在保险盖盖紧前请勿操作仪表。',
        'md_path': 'D:\\资料\\查重_简洁报告单\\auto\\查重_简洁报告单.md',
        'pdf_path': 'D:\\查重_简洁报告单.pdf',
        "chunks": [
            {
                "title": "查重_简洁报告单",
                "parent_title": "",
                "file_title": "查重_简洁报告单",
                "content": "查重_简洁报告单\n\n![RS PRO 品牌标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d329d008ba12d6f5eed073b52a378a6829cb4c1baef85b0d77934fa902bbb7fd.jpg)\n\n使用说明书\n\nRS-12\n\n编号: 123-1939\n\n数字万用表\n\n![中文操作界面标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/81735c16d6175e1dd624407b3448d8bf2039d8e22b1f9cc3941e533706a070a9.jpg)\n\nCE\n\n![RS-12数字万用表面板结构与功能标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/f179d9399297a15b5d4e764602734c25302eec0b528b231f0e455ca9c76dce0b.jpg)\n"
            },
            {
                "title": "## 安全手册",
                "parent_title": "## 安全手册",
                "file_title": "查重_简洁报告单",
                "content": "## 安全手册\n\n\n为了您的安全，请在使用本仪表之前仔细阅读该手册:\n\n使用本表时，请勿将输入的测量值超出其所允许的量程范围。\n\n\n\n- 【功能】：输入量程为最大输入。\n- 【交/直流电压】：输入量程为直流/交流电压600V。\n- 【直流/交流电压】：输入量程为直流/交流电压600V, 200Vrms 用于200mV量程。\n- 【mA直流】：输入量程为200mA 250V快速熔断保险丝。\n- 【A DC】：输入量程为10A 250V 快速熔断保险丝(最多每15分钟，需时30秒)。\n- 【电阻,短路测试】：输入量程为250Vrms, 最多15秒。\n\n\n\n2. 在测量高压电路时，请严格注意个人及设备的安全防护措施。\n\n3. 若负极端口（COM）电压超出500V以上接地电压，请勿进行电压测试。\n\n4. 若功能开关置于电流，电阻或二极管位置时，请勿将表笔与电路相连接，否则会损坏仪表。\n\n5. 进行电阻或二极管测试时，应把电容放电并断开电源。\n\n6. 打开后盖，更换保险丝或电池之前，请关闭电源并取下表笔。\n\n7. 请勿使用仪表，直到电池盖和保险丝盖装好，螺丝拧紧。\n"
            },
            {
                "title": "## 安全标识",
                "parent_title": "## 安全标识",
                "file_title": "查重_简洁报告单",
                "content": "## 安全标识\n\n\n![通用安全警告标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6ed1c2d2c192fe7422f77d0eb13133a4f4b01ea3e738ce2017bc5df75185e1a0.jpg)\n\n表明此操作须参照说明书进行。\n\nWARNING 表明此处可能出现危险电压，请避开以免导致死亡或严重伤害。\n\nCAUTION 表明此处可能出现危险电压，请避开以免导致仪表的损坏。\n\n![最大值标识（MAX）警示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/4033d9da04e1ffceb8382efdf1c281a8fbddf7ebe238b3aa131ff2f0c43fbeb0.jpg)\n\n请勿连接到500VAC或VDC的电路上。\n\n![闪电警示符号：表示危险电压，需避免接触以防致命伤害或设备损坏](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/2219ca75130874ec766983013be86c7afd233a4a1b3a5188990261156f1f2cc5.jpg)\n\n表明此端口可能出现危险电压。\n\n![双绝缘保护标识](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/6b7cc68242e90cec872c128c48320467c4ea101b9973420a5a2a2c46c1a7d489.jpg)\n\n双绝缘保护。\n"
            },
            {
                "title": "## 控制与端口",
                "parent_title": "## 控制与端口",
                "file_title": "查重_简洁报告单",
                "content": "## 控制与端口\n\n\n1.LCD液晶显示\n\n2.功能选择转盘\n\n3.10A端口\n\n4.COM端口\n\n5.正极端口\n\n6.数据保持按键\n\n7.背光按键\n\n![万用表各部件标识图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/d6946861c4592804bd8d7e75b58029565712d4dc58f855e374bf0fcf370c91dd.jpg)\n"
            },
            {
                "title": "## 功能符号指示",
                "parent_title": "## 功能符号指示",
                "file_title": "查重_简洁报告单",
                "content": "## 功能符号指示\n\n\n•))) 蜂鸣指示\n\n![二极管测试指示符号](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/e92ffd955b1ca1fd14290da681a763771c958cbcf0a73a332107f471f96c29b2.jpg)\n\n二极管测试指示\n\nµ micro (电流范围)\n\nm milli ( 电压/电流范围)\n\nk kilo (电阻范围)\n\nVDC 直流电压\n\nVAC 交流电流\n\nADC 直流电流\n\nBAT 电池电量不足指示\n"
            },
            {
                "title": "## 规格",
                "parent_title": "## 规格",
                "file_title": "查重_简洁报告单",
                "content": "## 规格\n\n## 规格-1 - 【直流电压】(对应功能)：量程为200mV，分辨率为0.1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为2000mV，分辨率为1mV，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为20V，分辨率为0.01V，精确度为± (0.5% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为± (0.8% reading + 2 digits)。\n- 【直流电压】(对应功能)：量程为600V，分辨率为1V，精确度为± (0.8% reading + 2 digits)。\n- 【交流电压】(对应功能)：量程为200V，分辨率为0.1V，精确度为± (1.2% reading + 10 digits50/60Hz)。\n- 【交流电压】(对应功能)：量程为600V，分辨率为1V，精确度为± (1.2% reading + 10 digits50/60Hz)。\n- 【直流电流】(对应功能)：量程为2000μA，分辨率为1μA，精确度为± (1.0% reading + 2 digits)。\n- 【直流电流】(对应功能)：量程为20mA，分辨率为10μA，精确度为± (1.0% reading + 2 digits)。\n- 【直流电流】(对应功能)：量程为200mA，分辨率为100μA，精确度为± (1.2% reading + 2 digits)。\n- 【直流电流】(对应功能)：量程为10A，分辨率为10mA，精确度为± (2.0% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200Ω，分辨率为0.1Ω，精确度为± (0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为2000Ω，分辨率为1Ω，精确度为± (0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为20kΩ，分辨率为0.01kΩ，精确度为± (0.8% reading + 2 digits)。\n- 【电阻】(对应功能)：量程为200kΩ，分辨率为0.1kΩ，精确度为± (0.8% reading + 2 digits)。"
            },
            {
                "title": "## 规格",
                "parent_title": "## 规格",
                "file_title": "查重_简洁报告单",
                "content": "## 规格\n\n## 规格-2 - 【电阻】(对应功能)：量程为2000kΩ，分辨率为1kΩ，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为9V，分辨率为10mV，精确度为± (1.0% reading + 2 digits)。\n- 【电池】(对应功能)：量程为1.5V，分辨率为1mV，精确度为± (1.0% reading + 2 digits)。\n\n## 规格-3 注意: 精确度规格由两种因素组成。  \n● (% reading) –测量电路的精确度。  \n● (+ digits) –数位转换器条码的精确度。  \n注意: 精确度在65°F 至 83°F (18°C 至 28°C)，湿度低于75%RH时得出。"
            },
            {
                "title": "## 技术指标说明",
                "parent_title": "## 技术指标说明",
                "file_title": "查重_简洁报告单",
                "content": "## 技术指标说明\n\n\n二极管测试 测试电流最大值1mA, 开路电压 2.8V DC典型值\n\n短路蜂鸣测试 若电阻小于30时产生蜂鸣\n\n电池测试电流 9V (6mA)；1.5V (100mA)\n\n输入阻抗 >1MΩ\n\n交流电压频宽 45Hz～450Hz\n\nDCA电压跌路测试 200mV\n\n显示 3 ½ 数位，2000位液晶显示，1.1”数位\n\n超量程提示 以“1”表示\n\n极性 自动(正极无显示);负极显示(-)\n\n测量率 正常情况下每秒2次\n\n低电池提示 电池电压不足时，显示BAT符号\n\n电池 一粒9V (NEDA 1604) 电池\n\n保险丝 mA, µA 量程;0.2A/250V 快速熔断保险丝，A 档量程10A/250V快速熔断保险丝\n\n操作环境 32°F～122°F (0°C～50°C)\n\n储存温度 -4°F～140°F (-20°C～60°C)\n\n相对湿度 <70% 操作, <80% 储存\n\n室内使用,最高海拔 7000英尺(2000米)\n\n重量 255g\n\n尺寸 150mm x 70mm x 48mm\n\n安全认证 室内使用，符合过电压类别II\n\n污染级别 2\n"
            },
            {
                "title": "## 电池安装",
                "parent_title": "## 电池安装",
                "file_title": "查重_简洁报告单",
                "content": "## 电池安装\n\n\n警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 把表笔与仪表断开。\n\n2. 用螺丝刀拧开电池后盖上的螺母。\n\n3. 正确安装电池，正负极应一致。\n\n4. 盖上电池后盖并拧紧螺丝钉。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n\n注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。\n"
            },
            {
                "title": "## 数值保持按键HOLD",
                "parent_title": "## 数值保持按键HOLD",
                "file_title": "查重_简洁报告单",
                "content": "## 数值保持按键HOLD\n\n\n保持键允许仪表固定测量值以供参考：\n\n1. 按下“HOLD”键保持读数， 同时出现“HOLD”字符\n\n2. 再次按下“DATA HOLD”键 切换至正常操作\n"
            },
            {
                "title": "## 背光灯键（BACKLIGHT）",
                "parent_title": "## 背光灯键（BACKLIGHT）",
                "file_title": "查重_简洁报告单",
                "content": "## 背光灯键（BACKLIGHT）\n\n\n1. 按下背光灯键开启背光灯。\n\n2. 再次按背光灯键关闭背光灯。\n\n警告：小心触电，高压电流十分危险，应小心操作。\n\n1. 为了节省电池损耗，使用后请将旋钮调至“OFF”档。\n\n2. 若测量过程中显示屏出现“OL”，表明测量值超出所选档位，应改选更高档。\n\n注意:在某些低交直流电压档位内，若表笔与被测物断开，显示屏将出现任意不稳定数值。该现象由高输入灵敏度所致。若接通电路，可读到稳定准确的数值。\n"
            },
            {
                "title": "## 测量非接触交流电压",
                "parent_title": "## 测量非接触交流电压",
                "file_title": "查重_简洁报告单",
                "content": "## 测量非接触交流电压\n\n\n警告: 为了防止电击，请在使用前，确保正确使用此非接触交流电压测电笔。\n\n1. 让其探头靠近或插入火线的输出插座孔时。\n\n2. 如果火线带有220V交流电输出，指示灯就会被点亮。\n\n注意: 如果是零线和火线缠绕在一起时，此时测试要将两线分开，来进行火线与零线的区分。\n\n注意: 此非接触交流电压测电笔设计为高度灵敏探测.当遇到静电或其它能带电体时，可能指示灯也会亮起或瞬间闪烁，这属于正常现象。\n"
            },
            {
                "title": "## 直流电压测量",
                "parent_title": "## 直流电压测量",
                "file_title": "查重_简洁报告单",
                "content": "## 直流电压测量\n\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n1. 将功能转盘置于V DC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物,确保极性正确(红色连正极,黑色连负极)。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值。若极性颠倒，数值前将显示负号。\n"
            },
            {
                "title": "## 交流电压测量",
                "parent_title": "## 交流电压测量",
                "file_title": "查重_简洁报告单",
                "content": "## 交流电压测量\n\n\n警告：谨防触电。\n\n若表笔长度不够不能接触到某些240V用具插座的带电部位，则可能出现插座有电而读到的数值却为0的情况。因此若无电压显示，应检查表笔是否接触到了插座内的金属接口。\n\n注意：正打开或关闭电源时不要进行此项测量，瞬间的强大电压将损坏仪表。\n\n![交流电压测量操作示意图（表笔连接与读数显示）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/84c37b209829d15820d5bbe76bbc98e1bf9eddc58bd9c983fc710cb2747d341b.jpg)\n\n1. 将功能转盘置于V AC的位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极V端口。\n\n3. 将表笔尖端接触被测物。\n\n4. 显示屏上读取电压值。显示屏显示了精确的小数点，数值和(AC,V等)符号。\n\n在显示屏上读取电压数据。不断重调功能转盘至低交流电压档位获得高分辨率读数。读数由精确的小数点和数值表示。\n"
            },
            {
                "title": "## 直流电流测量",
                "parent_title": "## 直流电流测量",
                "file_title": "查重_简洁报告单",
                "content": "## 直流电流测量\n\n\n注意：在10A情况下测量时间不能超过30秒，否则将可能损坏仪表或表笔。\n\n![直流电流测量接线示意图（10A档位）](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/8eb1e59b1e3f5e200f6d947da47dcd767fe061b91e72b1ba5325869677dcdad2.jpg)\n\n1. 将黑色表笔插入负极COM端口。\n\n2. 测量直流200mA 以下的电流,将功能转盘置于最高DC mA档位，并将红色表笔插入mA端口。\n\n3. 测量直流10A时,将功能转盘置于10A档位，并将红色表笔(10A)端口。\n\n4. 断开被测电路的电源。在你想测量电流的位置打开电路绝缘层。\n\n5. 将黑色表笔接触被测电路的负极，红色表笔接触被测电路正极。\n\n6. 接通电源。\n\n7. 在显示屏上读取读数。进行mA DC测量时,不断重调功能转盘至低mA DC档位获得高分辨率读数.读数由精确的小数点和数值表示。\n\n![直流10A电流测量接线示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/950918b0a12b239de83d45093fa5e6258bfa3d33848f927ecf0993f3210bf3d9.jpg)\n"
            },
            {
                "title": "## 电阻测量",
                "parent_title": "## 电阻测量",
                "file_title": "查重_简洁报告单",
                "content": "## 电阻测量\n\n\n警告: 为防触电,测量前应断开电源，把所有电容放电，取出电池和拔掉电线。\n\n1. 将功能转盘置于最高电阻Ω位置.\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口\n\n3. 把表笔接触被测电路或元件。测试时最好断开电路的一端，以使剩余的电路不会干扰被测电阻数值。\n\n4. 读取显示屏上读数，然后将功能转盘调至最低电阻Ω档位，通常大于实际电阻或预测电阻.读数由精确的小数点和数值表示。\n\n![数字万用表电阻测量操作示意图](http://106.75.224.144:9000/knowledge-base-files/查重_简洁报告单/dfbcdd205c8748df2005169dfc3c1b55f16dfe3a15024197c9d1a6b0064a9d6e.jpg)\n"
            },
            {
                "title": "## 短路蜂鸣测试",
                "parent_title": "## 短路蜂鸣测试",
                "file_title": "查重_简洁报告单",
                "content": "## 短路蜂鸣测试\n\n\n警告：请不要在接通电源的情况下进行在线短路蜂鸣测试以免触电。\n\n1. 将功能键转盘置于 位置。\n\n2. 将黑色表笔插入负极COM端口，红色表笔插入正极Ω端口。\n\n3. 把表笔与被测物体相接触。\n\n4. 当电阻小于30时Ω，仪表会发出蜂鸣.如果是开路，显示屏将显示“1”字符。\n"
            },
            {
                "title": "## 二极管测试",
                "parent_title": "## 二极管测试",
                "file_title": "查重_简洁报告单",
                "content": "## 二极管测试\n\n\n1. 将黑色表笔插入负极COM 端口，红色表笔插入正极V端口。\n\n2. 将功能转盘置于 位置。\n\n3. 把表笔与二极管相接触，正向电压将显示400 至 700mV.反向电压显示“ 1”符号.短路时将显示接近 0V，开路时会在两种极性上显示“1”符号。\n"
            },
            {
                "title": "## 电池测试",
                "parent_title": "## 电池测试",
                "file_title": "查重_简洁报告单",
                "content": "## 电池测试\n\n\n1. 将黑色表笔插入负极COM端口，红色表笔插入正极V 端口。\n\n2. 使用功能选择键，选择1.5V 或 9V 电池档位。\n\n3. 将红色表笔接触电池正极，将黑色表笔接触电池负极。\n\n4. 在显示屏上读取数值。\n\n\n\n- 【9V 电池：】：良好为>8.2V，较弱为7.2 至 8.2V，坏的为<7.2V。\n- 【1.5V 电池：】：良好为>1.35V，较弱为1.22 至 1.35V，坏的为<1.22V。\n\n\n"
            },
            {
                "title": "## 更换电池",
                "parent_title": "## 更换电池",
                "file_title": "查重_简洁报告单",
                "content": "## 更换电池\n\n\n警告：为防触电，打开电池后盖前后，请勿操作仪表并把表笔与电源断开。\n\n1. 当电池电压不足时，显示屏上会出现“BAT”符号，此时应更换电池。\n\n2. 按下面的步骤安装电池。\n\n3. 妥善处理废电池。\n\n警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。\n"
            },
            {
                "title": "## 更换保险丝",
                "parent_title": "## 更换保险丝",
                "file_title": "查重_简洁报告单",
                "content": "## 更换保险丝\n\n\n警告:为防触电，在打开保险丝门之前，请把表笔和电源断开。\n\n1. 把表笔与仪表及其它被测物断开。\n\n2. 用螺丝刀拧开保险丝门上的螺母。\n\n3. 轻轻取出废旧的保险丝。\n\n4. 装入新的保险丝。\n\n5. 使用正确型号与数值的保险丝(0.2A/250V) 快速熔断保险丝用于200mA的量程，10A/250V 快速熔断保险丝用于10A的量程。\n\n6. 盖回后盖，拧紧螺钉。\n\n警告: 为防触电，在保险盖盖紧前请勿操作仪表。"
            }
        ]
    }
    state = create_default_state(**init)
    node = BgeEmbeddingChunksNode()
    print(node(state))