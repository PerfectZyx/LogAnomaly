## 环境：
	python3, pygraphviz

## ft-tree.py：
* 输出文件：模板、单词词频列表
	* 运行脚本的的命令：
		* python ft\_tree.py -FIRST\_COL 0 -NO\_CUTTING 1 -CUTTING\_PERCENT 0.3 -data\_path ./bgl.log -template_path ./bgl.template -fre\_word\_path ./bgl.fre -leaf\_num 6 -short\_threshold 5 
	* 参数样例：
	    *  FIRST\_COL 每行日志从第几列作为输入，默认为0
	    *  NO\_CUTTING = 0 #初步设定1时，是前30% 不剪枝 ,全局开关， 当其为0时，全局按照min_threshold剪枝
	    *  CUTTING\_PERCENT =0.6 #前百分之多少是不剪枝的 
	    *  data\_path='input.txt'
	    *  template\_path = "./logTemplate.txt" #模板
	    *  fre\_word\_path = "./fre_word.txt"   
	    *  leaf\_num = 4 #剪枝数
	    *  short\_threshold = 2 #过滤掉长度小于5的日志

	
## matchTemplate.py:
* 运行脚本的的命令：
	* python matchTemplate.py -short\_threshold 5 -leaf\_num 6 -template\_path ./bgl.template -fre\_word\_path ./bgl.fre -log\_path ./bgl.log -out\_seq\_path ./bgl.seq -CUTTING\_PERCENT 0.3 -NO\_CUTTING 1 -match\_model 1
		
* 参数样例：
	*  short\_threshold = 5 #过滤掉长度小于5的日志
	*  leaf\_num 增量学习时的剪枝阈值。
	*  template\_path = './bgl.template'
	*  fre\_word\_path = './bgl.fre'
	*  log\_path = './bgl.log'
	*  out\_seq\_path = './bgl.seq'
	*  CUTTING\_PERCENT 指定每条日志的前百分之几的单词不剪枝，增量学习时会用到，正常匹配用不到
	*  NO\_CUTTING 是否每条日志的前几个单词不剪枝，0为正常剪枝，1为不剪枝，默认为1。增量学习时会用到，正常匹配用不到
	*  match\_model 1:正常匹配日志  2:单条增量学习&匹配 3:批量增量学习&匹配
* 增量学习模板：
	* matchLogsAndLearnTemplateOneByOne()函数  单条匹配，如果匹配不到，则学习新的模板。会将新学到的模板插入到模板文件的最后。
	* matchLogsFromFile() 函数，正常匹配日志，如果匹配不到，则为模板序号为0
	* LearnTemplateByIntervals(）函数， 将一时段的日志作为输入，基于以前的模板增量学习，新添加的日志模板也会按照设定的阈值剪枝，最终将新学到的模板插入到模板文件的最后。
