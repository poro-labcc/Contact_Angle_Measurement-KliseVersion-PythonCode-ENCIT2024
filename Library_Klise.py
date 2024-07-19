import numpy as np

#Mede angulo em relação ao fluido 2



def Klise(npimg, d):
    P = rastrear3d(npimg)
    c1, c2 = calcularplano3d(P, d)    
    dphi, dpsi = Derivadapontosrastreados(npimg, P)
    ang = angulocorrigido(c1, c2, dphi, dpsi)    
    Angles_position = Angleposition(ang)
    Angles = ang[ang != 0]
    return Angles, Angles_position

def Angleposition(array):
    non_zero_indices = np.transpose(np.nonzero(array))
    non_zero_values = array[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    combined_array = np.column_stack((non_zero_values, non_zero_indices))
    return combined_array

def rastrear3d(D):
    X, Y, Z = np.shape(D)
    P=np.zeros((X,Y,Z),dtype="uint8")
    for i in range(1, X-1):
        for j in range (1, Y-1):
            for k in range (1, Z-1):
                if D[i,j,k] == 2:
                    if D[i+1,j,k]==0 or D[i-1,j,k]==0 or D[i,j+1,k]==0 or D[i,j-1,k]==0 or D[i,j,k+1]==0 or D[i,j,k-1]==0:
                        P[i,j,k] = 1
                    if D[i+1,j,k]==1 or D[i-1,j,k]==1 or D[i,j+1,k]==1 or D[i,j-1,k]==1 or D[i,j,k+1]==1 or D[i,j,k-1]==1:
                        P[i,j,k] = 2
                        if D[i+1,j,k]==0 or D[i-1,j,k]==0 or D[i,j+1,k]==0 or D[i,j-1,k]==0 or D[i,j,k+1]==0 or D[i,j,k-1]==0:
                            P[i,j,k] = 3
    return P;

def encontrarpontosconectados3d(P, d, M, N, O, fl1, fl2):
    #pocura esferico, não quadrado
    conectados = np.zeros((1,3),dtype='int')
    contador = 1
    c = 0
    while c < contador:
        for i in range(-1+conectados[c,0], 2+conectados[c,0]):
            for j in range(-1+conectados[c,1], 2+conectados[c,1]):
                for k in range(-1+conectados[c,2], 2+conectados[c,2]):
                    if P[M+i, N+j, O+k] == fl1 or P[M+i, N+j, O+k] == fl2:
                        teste = 0
                        for h in range(0, contador):
                            if i == conectados[h,0] and j == conectados[h,1] and k == conectados[h,2]:
                                teste = 1
#                        if i > d or -i > d or j > d or -j > d or k > d or -k > d:
#                            teste = 1
                        if i**2 + j**2 + k**2 > d**2:
                            teste = 1
                        if teste == 0:
                            conectados = np.lib.pad(conectados, ((0, 1), (0, 0)), 'constant', constant_values=(0))
                            conectados[contador, 0] = i
                            conectados[contador, 1] = j
                            conectados[contador, 2] = k
                            contador = contador + 1
        c = c + 1
    return conectados;

def melhoreixo3d(v):
    vn = np.unique(v[:,[1,2]], axis=0)
    a, w = np.shape(vn)
    vn = np.unique(v[:,[0,2]], axis=0)
    b, w = np.shape(vn)
    vn = np.unique(v[:,[0,1]], axis=0)
    c, w = np.shape(vn)
    if a > b and a > c:
        eixo = 0
    elif b > a and b > c:
        eixo = 1
    elif c > a and c > b:
        eixo = 2
    elif a == b > c:
        if v[0,0] == v[1,0]:
            eixo = 1
        else:
            eixo = 0
    elif a == c > b:
        if v[0,0] == v[1,0]:
            eixo = 2
        else:
            eixo = 0
    elif b == c > a:
        if v[0,1] == v[1,1]:
            eixo = 2
        else:
            eixo = 1
    elif a == b == c:
        eixo = 0
    return eixo;

def calcularplano3d(P, d):
    X, Y, Z = np.shape(P)
    coeficientes1=np.zeros((X,Y,Z,4),dtype=float)
    coeficientes2=np.zeros((X,Y,Z,4),dtype=float)
    for i in range(d,X-d):
        for j in range (d, Y-d):
            for k in range(d, Z-d):
                if P[i,j,k] == 3:  
                    v1 = encontrarpontosconectados3d(P, d, i, j, k, 1, 3)
                    v2 = encontrarpontosconectados3d(P, d, i, j, k, 2, 3)            
                    if np.size(v1) > 21 and np.size(v2) > 21:
                        eixo1 = melhoreixo3d(v1)
                        eixo2 = melhoreixo3d(v2)
                        matriz_design1 = np.c_[np.delete(v1, eixo1, axis=1), np.ones(v1.shape[0])]
                        C, _, _, _ = np.linalg.lstsq(matriz_design1, v1[:, eixo1], rcond=None)
                        coeficientes1[i,j,k,eixo1] = -1
                        coeficientes1[i,j,k,3] = C[2]
                        if eixo1 == 0:
                            coeficientes1[i,j,k,1] = C[0]
                            coeficientes1[i,j,k,2] = C[1]
                        if eixo1 == 1:
                            coeficientes1[i,j,k,0] = C[0]
                            coeficientes1[i,j,k,2] = C[1]
                        if eixo1 == 2:
                            coeficientes1[i,j,k,0] = C[0]
                            coeficientes1[i,j,k,1] = C[1]
                        matriz_design2 = np.c_[np.delete(v2, eixo2, axis=1), np.ones(v2.shape[0])]
                        C2, _, _, _ = np.linalg.lstsq(matriz_design2, v2[:, eixo2], rcond=None)
                        coeficientes2[i,j,k,eixo2] = -1
                        coeficientes2[i,j,k,3] = C2[2]
                        if eixo2 == 0:
                            coeficientes2[i,j,k,1] = C2[0]
                            coeficientes2[i,j,k,2] = C2[1]
                        if eixo2 == 1:
                            coeficientes2[i,j,k,0] = C2[0]
                            coeficientes2[i,j,k,2] = C2[1]
                        if eixo2 == 2:
                            coeficientes2[i,j,k,0] = C2[0]
                            coeficientes2[i,j,k,1] = C2[1]
    return coeficientes1, coeficientes2;

def gphi(npimg):
    M, N, O = np.shape(npimg)
    phi = np.zeros((M, N, O), dtype="float")
    for i in range(0,M):
        for j in range (0, N):
            for k in range (0, O):
                if npimg[i,j,k] == 2 or npimg[i,j,k] == 0:
                    phi[i,j,k] = -1
                if npimg[i,j,k] == 1:
                    phi[i,j,k] = 1
    return phi;


def gpsi(npimg):
    M, N, O = np.shape(npimg)
    psi = np.zeros((M, N, O), dtype="float")
    for i in range(0,M):
        for j in range (0, N):
            for k in range (0, O):
                if npimg[i,j,k] == 0:
                    psi[i,j,k] = -1
                if npimg[i,j,k] == 1 or npimg[i,j,k] == 2:
                    psi[i,j,k] = 1
    return psi;



def juntarmetrizesvetor(mt1, mt2, mt3):
    M,N,O = np.shape(mt1)
    mt = np.zeros((M,N,O,3))
    mt[:,:,:,0] = mt1
    mt[:,:,:,1] = mt2
    mt[:,:,:,2] = mt3
    return mt;

def Derivadapontosrastreados(D, P):
    x, y, z = np.shape(D)
    phi = gphi(D)
    psi = gpsi(D)
    dxphi = np.zeros(np.shape(phi), dtype="float")
    dyphi = np.zeros(np.shape(phi), dtype="float")
    dzphi = np.zeros(np.shape(phi), dtype="float")
    dxpsi = np.zeros(np.shape(phi), dtype="float")
    dypsi = np.zeros(np.shape(phi), dtype="float")
    dzpsi = np.zeros(np.shape(phi), dtype="float")
    for i in range(0,x-1):
        for j in range (0, y-1):
            for k in range (0, z-1):            
                if P[i,j,k] == 3:
                    dxphi[i,j,k] = (int(phi[i+1, j,k])-int(phi[i-1, j, k]))/2
                    dyphi[i,j,k] = (int(phi[i, j+1,k])-int(phi[i, j-1, k]))/2
                    dzphi[i,j,k] = (int(phi[i, j,k+1])-int(phi[i, j, k-1]))/2
                    dxpsi[i,j,k] = (int(psi[i+1, j,k])-int(psi[i-1, j, k]))/2
                    dypsi[i,j,k] = (int(psi[i, j+1,k])-int(psi[i, j-1, k]))/2
                    dzpsi[i,j,k] = (int(psi[i, j,k+1])-int(psi[i, j, k-1]))/2  
    dphi =  juntarmetrizesvetor(dxphi, dyphi, dzphi)
    dpsi =  juntarmetrizesvetor(dxpsi, dypsi, dzpsi)
    return dphi, dpsi;

def vetorunitario(C):
    I, J, K, L = np.shape(C)
    cv = np.zeros((I,J,K), dtype="float")
    mtuni = np.zeros((I,J,K,L), dtype="float")
    for i in range(0,I):
        for j in range (0, J):
            for k in range (0, K):
                cv[i,j,k] = np.sqrt((C[i,j,k,0])**2+(C[i,j,k,1])**2 + C[i,j,k,2]**2) 
                if cv[i,j,k] != 0:
                    mtuni[i,j,k,:] = (C[i,j,k,:])/(cv[i,j,k])
    return mtuni;

def angulocorrigido(c1, c2, dphi, dpsi):
    I, J, K, L = np.shape(c1)
    ang = np.zeros((I, J, K),dtype=float)
    c1 = vetorunitario(c1)
    c2 = vetorunitario(c2)
    dphi = vetorunitario(dphi)
    dpsi = vetorunitario(dpsi)
    for i in range(0, I):
        for j in range(0, J):
            for k in range(0, K):               
                a = c1[i,j,k,0]*dpsi[i,j,k,0] + c1[i,j,k,1]*dpsi[i,j,k,1] + c1[i,j,k,2]*dpsi[i,j,k,2]
                if a < -1:
                    a = -1
                if a > 1:
                    a = 1
                b = c2[i,j,k,0]*dphi[i,j,k,0] + c2[i,j,k,1]*dphi[i,j,k,1] + c2[i,j,k,2]*dphi[i,j,k,2]
                if b < -1:
                    b = -1
                if b > 1:
                    b = 1
                if np.arccos(a) >= np.pi/2:
                    c1[i,j,k,:] = -1*c1[i,j,k,:]
                if np.arccos(b) >= np.pi/2:
                    c2[i,j,k,:] = -1*c2[i,j,k,:]
                ang[i,j,k] = c1[i,j,k,0]*c2[i,j,k,0] + c1[i,j,k,1]*c2[i,j,k,1] + c1[i,j,k,2]*c2[i,j,k,2]
                if ang[i,j,k] > 1:
                    ang[i,j,k] = 1
                if ang[i,j,k] < -1:
                    ang[i,j,k] = -1
                if ang[i,j,k] != 0:
                    #all(l != 0 for l in c1[i,j,k,:]) and all(l != 0 for l in c1[i,j,k,:]) and 
                    ang[i,j,k] = np.arccos((ang[i,j,k]))
                    ang[i,j,k] = ang[i,j,k]
    return ang;
