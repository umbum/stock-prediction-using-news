1.Get two version of python, 32bit and 64bit.
	64bit python for tensorflow.
	32bit python for CYBOS, pywin23.
	
	ex) In anaconda prompt,
		$set CONDA_FORCE_32BIT=1
		$conda create -n (name of env) python=(python version)
	Then, you got 32bit python vertual environment.
	
	$deactivate 
	or
	$set CONDA_FORCE_32BIT=0

	will turn off 32 option.

	To get requirements,
	64 bit python env: $pip install -r requirements.txt
	32 bit python env: $pip install -r requirements2.txt



2. To get TA-Lib installed,
	*It works with CYBOS which only works on python 32bit. ==> Get python 32bit!

	https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

	Get wheel file and move to working directory.

	If your python version is 3.6,
	$pip install TA_Lib-0.4.17-cp36-cp36m-win32.whl


3. To get 대신증권 api a.k.a CYBOS, 
	*It only works on python 32bit.

	Make your acount, at below link.
	https://www.daishin.com/g.ds?m=4027&p=3979&v=2983

	Or, if you not made regular member acount(ex. web acount), go to below link.
	http://vt.daishin.com/ds/cybos/info/info.do 

	and access menu '회원가입 바로가기' to create acount for simulated investment, 
	and access '모의투자 참가신청' to apply.

	Then, download CYBOS 5 from below link,
	http://money2.daishin.com/E5/WTS/Customer/GuideTrading/DW_DownloadCenter.aspx?m=1101&p=2669&v=2248&gclid=Cj0KCQiAnOzSBRDGARIsAL-mUB2eT8Di250ZRZK-JSJ9W1DtiRc5Keg99_-WlWFFfAhCfv4BkU-81SQaAuESEALw_wcB

	Log in with your acount, in menu CYBOS PLUS with '모의투자'(simulated investment) option.