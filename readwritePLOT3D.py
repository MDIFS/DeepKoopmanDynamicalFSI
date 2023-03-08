import numpy as np

# http://ig.hateblo.jp/entry/2014/05/30/225607
# https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html

def checkheader(fname): 
    fd = open(fname,"r")
    head, tail = ("head",">i8"), ("tail",">i8") # ">:big-endian,<:little-endian"
    dtp = np.dtype([head, ("jmax",">i4"), ("kmax",">i4"), ("lmax",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    if chunk[0] == 12:
        iheader = 8
    else:
        iheader = 4
    fd.close()
    return iheader

def readgrid(fname,iheader,iprec): # 8/4 = double/single precision
    fd = open(fname,"r")
    head, tail = ("head",">i"+str(iheader)), ("tail",">i"+str(iheader)) # ">:big-endian,<:little-endian"
    dtp = np.dtype([head, ("jmax",">i4"), ("kmax",">i4"), ("lmax",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    jmax,kmax,lmax = chunk["jmax"],chunk["kmax"],chunk["lmax"]
    dtp = np.dtype([head, ("r",">"+str(jmax*kmax*lmax*3)+"f"+str(iprec)), tail]) # assume double precision
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    r = chunk["r"].reshape((jmax,kmax,lmax,3),order="F")
    #
    head,tail = ('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    dtp = ( [head,('js','>i4'),('je','>i4'),('ks','>i4'),('ke','>i4')
             ,('ls','>i4'),('le','>i4'),('ite1','>i4'),('ite2','>i4')
             ,('jd','>i4'),('imove','>i4'),tail] )
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    js,je,ks,ke = chunk['js'],chunk['je'],chunk['ks'],chunk['ke'] 
    ls,le,ite1,ite2 = chunk['ls'],chunk['le'],chunk['ite1'],chunk['ite2']
    jd,imove = chunk['jd'],chunk['imove'] 
    ibottom = [0]*10
    ibottom = js,je,ks,ke,ls,le,ite1,ite2,jd,imove 
    #
    fd.close()
    return r, ibottom

def writegrid(fname,r,iheader,ibottom,iprec): # 8/4 = double/single precision
    fd = open(fname,'wb')
    headn,tailn,head,tail = 4*3,4*3,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    jmax,kmax,lmax = r.shape[:3]
    dtp = [head,('jmax','>i4'),('kmax','>i4'),('lmax','>i4'),tail]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['jmax'],chunk[0]['kmax'],chunk[0]['lmax'] = jmax,kmax,lmax
    chunk.tofile(fd)
    headn,tailn = iprec*jmax*kmax*lmax*3,iprec*jmax*kmax*lmax*3
    dtp = [('head','>i'+str(iheader)),('r','>'+str(jmax*kmax*lmax*3)+'f'+str(iprec)),('tail','>i'+str(iheader))]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['r'] = np.array(r.reshape(jmax*kmax*lmax*3,order="F") )
    chunk.tofile(fd)
    #
    headn,tailn,head,tail = 4*10,4*10,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    js,je,ks,ke,ls,le,ite1,ite2,jd,imove = ibottom[:]
    dtp = ( [head,('js','>i4'),('je','>i4'),('ks','>i4'),('ke','>i4')
             ,('ls','>i4'),('le','>i4'),('ite1','>i4'),('ite2','>i4')
             ,('jd','>i4'),('imove','>i4'),tail] )
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['js'],chunk[0]['je'],chunk[0]['ks'],chunk[0]['ke'] = js,je,ks,ke
    chunk[0]['ls'],chunk[0]['le'],chunk[0]['ite1'],chunk[0]['ite2'] = ls,le,ite1,ite2
    chunk[0]['jd'],chunk[0]['imove'] = jd,imove
    chunk.tofile(fd)
    #
    fd.close()


def readflow(fname,iheader): # single precision
    fd = open(fname,"r")
    head, tail = ("head",">i"+str(iheader)), ("tail",">i"+str(iheader)) # ">:big-endian,<:little-endian"
    dtp = np.dtype([head, ("jmax",">i4"), ("kmax",">i4"), ("lmax",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    jmax,kmax,lmax = chunk["jmax"],chunk["kmax"],chunk["lmax"]
    dtp = np.dtype([head, ("fsmach",">f4"), ("alp",">f4"), ("totime",">f4"), ("nc",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    fsmach,alp,totime,nc = chunk["fsmach"],chunk["alp"],chunk["totime"],chunk["nc"]
    dtp = np.dtype([head, ("q",">"+str(jmax*kmax*lmax*5)+"f4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    q = chunk["q"].reshape((jmax,kmax,lmax,5),order="F")
    fd.close()
    return q,{"fsmach":fsmach,"alp":alp,"totime":totime,"nc":nc}

def readudvals(fname,iheader): # single precision
    fd = open(fname,"r")
    head, tail = ("head",">i"+str(iheader)), ("tail",">i"+str(iheader)) # ">:big-endian,<:little-endian"
    dtp = np.dtype([head, ("jmax",">i4"), ("kmax",">i4"), ("lmax",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    jmax,kmax,lmax = chunk["jmax"],chunk["kmax"],chunk["lmax"]
    dtp = np.dtype([head, ("isn",">i4"), ("ivn",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    isn,ivn = chunk["isn"],chunk["ivn"]
    q = np.zeros((jmax,kmax,lmax,isn))
    for isnum in range(isn):
        dtp = np.dtype([head, ("q",">"+str(jmax*kmax*lmax)+"f8"), tail])
        chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
        q[:,:,:,isnum] = chunk["q"].reshape((jmax,kmax,lmax),order="F")
    fd.close()
    return q

def writeflow(fname,q,statedic,iheader): # single precision
    jmax,kmax,lmax = q.shape[:3]
    fd = open(fname,'wb')
    headn,tailn,head,tail = 4*3,4*3,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    dtp = [head,('jmax','>i4'),('kmax','>i4'),('lmax','>i4'),tail]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['jmax'],chunk[0]['kmax'],chunk[0]['lmax'] = jmax,kmax,lmax
    chunk.tofile(fd)
    headn,tailn,head,tail = 4*4,4*4,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    dtp = [head,('fsmach','>f4'),('alp','>f4'),('totime','>f4'),('nc','>i4'),tail]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['fsmach'],chunk[0]['alp'],chunk[0]['totime'],chunk[0]['nc'] \
        = statedic['fsmach'],statedic['alp'],statedic['totime'],statedic['nc']
    chunk.tofile(fd)
    headn,tailn = 4*jmax*kmax*lmax*5,4*jmax*kmax*lmax*5
    dtp = [('head','>i'+str(iheader)),('q','>'+str(jmax*kmax*lmax*5)+'f4'),('tail','>i'+str(iheader))]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['q'] = np.array(q.reshape(jmax*kmax*lmax*5,order="F") )
    chunk.tofile(fd)
    fd.close()


def writeudvals(fname,q,iheader): # single precision user defined functions
    jmax,kmax,lmax,isud = q.shape
    fd = open(fname,'wb')
    headn,tailn,head,tail = 4*3,4*3,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    dtp = [head,('jmax','>i4'),('kmax','>i4'),('lmax','>i4'),tail]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['jmax'],chunk[0]['kmax'],chunk[0]['lmax'] = jmax,kmax,lmax
    chunk.tofile(fd)
    headn,tailn,head,tail = 4*2,4*2,('head','>i'+str(iheader)),('tail','>i'+str(iheader))
    dtp = [head,('is','>i4'),('iv','>i4'),tail]
    chunk = np.zeros(1, dtype=dtp)
    chunk[0]['head'],chunk[0]['tail'] = headn,tailn
    chunk[0]['is'],chunk[0]['iv'] = isud,0
    chunk.tofile(fd)
    for isnum in range(isud):
        headn,tailn = 4*jmax*kmax*lmax,4*jmax*kmax*lmax
        dtp = [('head','>i'+str(iheader)),('q','>'+str(jmax*kmax*lmax)+'f4'),('tail','>i'+str(iheader))]
        chunk = np.zeros(1, dtype=dtp)
        chunk[0]['head'],chunk[0]['tail'] = headn,tailn
        chunk[0]['q'] = np.array(q[:,:,:,isnum].reshape(jmax*kmax*lmax,order="F") )
        chunk.tofile(fd)
    fd.close()

def readforce(fname,iheader, iprec): # 8/4 = double/single precision
    fd = open(fname,"r")
    head, tail = ("head",">i"+str(iheader)), ("tail",">i"+str(iheader)) # ">:big-endian,<:little-endian"
    dtp = np.dtype([head, ("jmax",">i4"), ("kmax",">i4"), ("lmax",">i4"), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    jmax,kmax,lmax = chunk["jmax"],chunk["kmax"],chunk["lmax"]
    dtp = np.dtype([head, ("fx",">"+str(jmax*kmax*lmax*1)+"f"+str(iprec)), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    fx = chunk["fx"].reshape((jmax,kmax,lmax,1),order="F")
    dtp = np.dtype([head, ("fy",">"+str(jmax*kmax*lmax*1)+"f"+str(iprec)), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    fy = chunk["fy"].reshape((jmax,kmax,lmax,1),order="F")
    dtp = np.dtype([head, ("fz",">"+str(jmax*kmax*lmax*1)+"f"+str(iprec)), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    fz = chunk["fz"].reshape((jmax,kmax,lmax,1),order="F")
    dtp = np.dtype([head, ("f",">"+str(jmax*kmax*lmax*3)+"f"+str(iprec)), tail])
    chunk = np.fromfile(fd, dtype=dtp, count=1)[0]
    f = chunk["f"].reshape((jmax,kmax,lmax,3),order="F")

    fd.close()

    return f


#jmaxf,kmaxf,lmaxf = chunk["jmax"],chunk["kmax"],chunk["lmax"]
#if jmax != jmaxf or  kmax != kmaxf or  lmax != lmaxf: raise Exception
#    if nstep != nc: raise Exception # time step


def qtoqp(q):
    gamma = 1.4
    gami = gamma-1.0
    rh = q[:,:,:,0]
    rhu = q[:,:,:,1]
    rhv = q[:,:,:,2]
    rhw = q[:,:,:,3]
    rhe = q[:,:,:,4]
    qp = np.zeros_like(q)
    u,v,w = rhu/rh,rhv/rh,rhw/rh
    ps = gami*(rhe - rh*0.5*(u*u + v*v + w*w))
    qp[:,:,:,0] = rh
    qp[:,:,:,1] = u
    qp[:,:,:,2] = v
    qp[:,:,:,3] = w
    qp[:,:,:,4] = ps*gamma # Originally normalized by sound speed
    return qp


def qptoq(qp):
    gamma = 1.4
    gami = gamma-1.0
    rh = qp[:,:,:,0]
    u = qp[:,:,:,1]
    v = qp[:,:,:,2]
    w = qp[:,:,:,3]
    ps = qp[:,:,:,4]/gamma # Originally normalized by sound speed
    q = np.zeros_like(qp)
    rhu,rhv,rhw = rh*u,rh*v,rh*w
    rhe = ps/gami + 0.5*rh*(u*u + v*v + w*w)
    q[:,:,:,0] = rh
    q[:,:,:,1] = rhu
    q[:,:,:,2] = rhv
    q[:,:,:,3] = rhw
    q[:,:,:,4] = rhe
    return q
