
META_INFO = {
"NLI":
"""自然语言推理任务中，输入包含两个文本：一个是前提，另一个是假设，任务的目标是判断这两个文本之间的逻辑关系，即假设是由前提蕴含、与前提矛盾还是与前提为中立。这一任务要求你不仅要理解每个文本的表面文字意义，还要深入理解文本间的隐含关系和上下文意义。""",


"SENTIMENT_2":
"""情感二分类任务旨在识别并分类文本中的情感倾向为积极或者消极。它要求你能够理解和解析给定文本的语义内容和情感色彩，依据文本所表达的态度、情绪或评价进行分类。成功的情感判别不仅能够识别明显的情感表达，如喜悦或悲伤的直接描述，还能捕捉到更微妙的情感暗示和语境中的情感色彩。""",

"smp2020-ewect": 
"""微博情绪六分类任务的文本来自于随机收集的包含各种话题的数据。该任务旨在识别并分类微博文本中蕴含的情绪为积极、愤怒、悲伤、恐惧、惊奇或者无情感。""",

"nlpcc-stance": 
"""微博文本立场判别任务给定一条微博以及它所评论的对象，该任务旨在识别并分类微博文本对于给定的评论对象的立场为支持、反对或者中立。
请注意，立场判别任务不仅要求你能够识别明显的立场的表达，如支持或者反对的直接描述，还需要你能够捕捉到更微妙隐含的立场暗示，揣摩评论人的心里活动，从而推断出它的立场。只有对于评论对象完全客观的描述才会被判定为中立。""",


"ASAP_ASPECT":
"""该评价对象级情感三分类任务源于针对饭店餐馆的评论。具体来说，给定一段关于饭店餐馆的评论和评论中的一个评价对象，通常是饭店餐馆的某个方面或属性，该任务旨在识别并判断评论对于该评价对象、方面或属性的情感倾向为中性、积极或者消极。
请注意，该任务关注的不是整体评论的情感色彩，而是要求你能够细化到评论中具体提及的各个方面或对象，针对给定的评价对象，判断评论对于该评价对象的情感。""",

"bdci2018":
"""该汽车评论情感三分类任务源于用户在汽车论坛中对汽车相关内容的讨论或评价。具体来说，给定一段对于汽车的评论和一个评价维度，通常是汽车的某个方面或属性，该任务旨在识别并判断评论从该评价维度出发，蕴含情感倾向为积极、消极还是中性。
请注意，该任务关注的不是整体评论的情感色彩，而是要求你能够细化到评论中具体提及的各个方面或对象，针对给定的评价维度，判断评论对于该维度所蕴含的情感倾向。""",



"lcqmc":
"""该问题对匹配任务来源于百度知道领域，旨在理解并比较两个中文问题的语义内容是否一致。输入包含两个问题（问题1和问题2），你需要判断这两个问题是否语义相似或表达相同的查询意图。如果两个问题在本质上询问的是相同的信息，那么他们的语义关系为匹配；否则，语义关系为不匹配。""",

"afqmc":
"""该文本语义相似判别任务中的文本来源于金融客服里的用户提问，旨在理解并比较两个问题所表述的语义内容是否一致。输入包含两个问题文本（文本1和文本2），你需要判断这两个问题是否语义相似或表达相同的疑问。如果两个问题在本质上询问的是相同的信息，那么他们之间的语义关系为匹配；否则，语义关系为不匹配。""",

"paws":
"""在文本对语义关系匹配任务中，你需要理解并且比较两个文本（文本1和文本2）是否具有相同的释义（含义）。如果两个文本的语义相似或者表达的意思相同，那么他们的语义关系为匹配；否则，语义关系为不匹配。
值得注意的是，该任务的特点是具有高度重叠词汇，重点需要你深入理解文本的句法结构。""",


"bustm":
"""该问题对语义关系任务来源于用户和智能助手之间的真实对话，旨在理解并判别两个用户提问之间的语义匹配关系。该任务的特点是文本较短、非常口语化、存在文本高度相似而语义不同的难例。输入包含两个问题（问题1和问题2），你需要判断这两个问题是否语义相似或者表达出接近的提问意图。如果两个问题在本质上询问的是相同的信息，那么他们的语义关系为匹配；否则，语义关系为不匹配。""",

"qbqtc":
"""该查询对相关性判别任务来源于搜索引擎中真实的用户查询，旨在理解并判别两个用户查询之间的相关性。输入包含两个查询（查询1和查询2），你需要判断这两个查询是否语义相似或者表达出接近的查询意图，从而输出相关程度差、有一定相关性、非常相关中的一个。""",

"cluewsc2020":
"""代词指代消歧任务旨在判断给定文本中的代词指代的是文中的哪个名词（或名词短语）。具体来说，给定一个含有代词的句子以及句子中的一个名词（或名词短语），任务是判断这个名词（或名词短语）是否为代词所指代的对象。""",



'cluener':
"""命名实体识别任务旨在识别文本中具有特定意义的实体，并将它们标注为其所属的类型。当前任务的实体包括：地址，书名，公司，游戏，政府，电影，姓名，组织机构，景点，职位。""",

'msra_ner':
"""命名实体识别任务旨在识别文本中具有特定意义的实体，并将它们标注为其所属的类型。当前任务的实体包括：地点，人物，组织。""",

'weibo_ner':
"""命名实体识别任务旨在识别文本中具有特定意义的实体，并将它们标注为其所属的类型。当前任务的实体包括：地点，人物，组织，地缘政治。""",


'tnews':
"""在新闻短文本分类任务中，你需要通过分析输入的新闻内容，输出该新闻所属的类别。此任务需要对新闻文本的主题、内容、关键词等多个方面进行综合理解，正确识别出新闻的主题和关键信息点，从而准确地归类到预先定义的类别中，包括：故事，文化，娱乐，体育，财经，家居，汽车，教育，科技，军事，旅行，世界，股票，农业，游戏。注意每个类别的界定不是完全孤立的，有些新闻可能涉及边界领域，需要根据新闻的主要内容和重点进行归类，每个新闻有且只属于上述一个类别。""",
'domain_cls':
"""在文本领域判别任务中，你需要分析文本内容，输出该文本最可能属于国民经济行业分类中的哪个类别，包括：交通运输仓储邮政、住宿餐饮、信息软件、农业、制造业、卫生医疗、国际组织、建筑、房地产、政府组织、教育、文体娱乐、水利环境、电力燃气水生产、科学技术、租赁法律、采矿、金融。每个文本有且只属于一个领域。""",
'csl':
"""在论文标题学科分类任务中，你需要通过分析给定的论文标题，根据标题推测其可能对应的论文内容，最终将该论文归类到它最可能属于的学科门类中，包括：法学，工学，经济学，理学，农学，管理学，医学，历史学，艺术学，哲学，教育学，军事学，文学。每个标题有且只属于一个学科。""",

'Chinese-Metaphor-Analysis':
"""中文动名词隐喻识别任务旨在识别文本中是不是使用了动词或者名词的隐喻。它要求你首先对动词及其关联的名词实体进行分析，判断文本是否使用了隐喻这一修辞手法，如果没有，输出没有隐喻。有的话，你需要进一步判别文本用到的是动词隐喻还是名词隐喻。""",

'crisis':
"""应急救援信息量判别任务的目的是确定是否给定的博文对于应急响应或应急救援有帮助。这里的应急响应包括抢救生命、减轻痛苦和重建家园等。具体来讲，可以是帮助无家可归者， 提供食物、水、住所、医疗等给受害者，维修道路、桥梁等关键基础设施等。根据博文中含有的信息量，你需要将博文分类为有用、无用或者不能判断。""",


'morality':
"""中文文本道德判别任务旨在识别并分类文本中蕴含的道德是正面还是负面的。""",

'cold':
"""中文文本歧视判别任务，给定一段文本和一个主题（种族、地域、性别中的一个），你需要判断文本中是否含有对于给定主题的歧视或者冒犯的含义。如果文本存在特定主题的歧视，输出是；否则，输出否。""",


'c3':
"""多项选择式阅读理解任务，给定一个背景(长文本或者对话)、一个问题和四个选项，你需要在理解背景的基础上，针对问题，从四个选项中选择出一个正确答案。""",

'logiqa':
"""逻辑问答领域的多项式阅读理解任务，给定一个背景文本，一个问题和四个选项，你需要在理解背景内容的基础上，对问题进行深入的逻辑分析和推理，以选择出最合适的答案。这一过程不仅要求深入理解背景文本，还需要运用批判性思维、逻辑推理能力，来分析各个选项与问题的相关性以及它们对文本的支持或削弱程度，从四个选项中选择出一个正确答案。""",
'logiqa2':
"""逻辑问答领域的多项式阅读理解任务，给定一个背景文本，一个问题和四个选项，你需要在理解背景内容的基础上，对问题进行深入的逻辑分析和推理，以选择出最合适的答案。这一过程不仅要求深入理解背景文本，还需要运用批判性思维、逻辑推理能力，来分析各个选项与问题的相关性以及它们对文本的支持或削弱程度，从四个选项中选择出一个正确答案。""",


'MRC':
"""片段抽取式阅读理解任务，给定一个问题和一个段落，你需要从段落中抽取连续的序列，使得该序列尽可能的作为该问题的答案。""",

"DuReader_yseorno":
"""观点极性判断阅读理解任务，给定一个问题和一个回答，你需要判断回答中所表述的是非观点极性。""",

"DuReader_checklist":
"""在片段抽取式阅读理解任务中，给定一个问题和一个段落，你首先需要判断该段落中是否包含给定问题的答案，如果是，你需要从段落中抽取连续的序列，使得该序列尽可能的作为该问题的答案；否则，你需要输出无答案。""",

"ReCO":
"""观点判别阅读理解任务，给定一个背景、一个问题和几个选项（代表不同观点），你需要在理解背景的基础上，针对问题，从选项中选择出正确的观点。""",

'ekar':
"""词语类比识别任务，给定一个问题（一组词语）和几个选项（每个选项也是一组词语），你首先需要分析问题中词语之间蕴含的关系，然后分析每个选项中词语之间蕴含的关系，最终从选项中选择出和问题中蕴含的关系最为一致的答案。
请注意，从某些角度来看，所有选项蕴含的关系都可能与问题的关系相关。你的挑战在于找到最相关的选项，即考虑语言特点、术语顺序、常识知识等属性，找出问题中的词语与选项之间的内在联系和关系。""",

'FCGEC':
"""在中文语法检错任务中，给定一个文本，你需要检查文本中是否包含语法错误。该任务中的语法错误的类型有：语序不当、搭配不当、成分缺失、成分赘余、结构混乱、不合逻辑、语意不明。如果没有语法错误，输出无错误。如果文本中存在语法错误，你需要检查出句子中存在哪个语法错误，并且输出文本包含的语法错误。
请注意，一个文本中最多存在一个类型的语法错误。""",

'NaCGEC':
"""在中文语法检错任务中，给定一个文本，你需要检查文本中是否包含语法错误。该任务中的语法错误的类型有：搭配不当、不合逻辑、语序不当、成分残缺、成分赘余、句式杂糅。如果没有语法错误，输出无错误。如果文本中存在语法错误，你需要检查出句子中存在哪个语法错误，并且输出文本中包含的语法错误。
请注意，一个文本中最多存在一个类型的语法错误。"""

}
