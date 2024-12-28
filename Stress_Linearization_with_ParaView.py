"""
Created on Wed Jun 15 14:48:43 2022
Author: Jakub Trušina
Name: Stress_Linearization_with_ParaView.py
"""

inp = "SIGM_from_ParaView_Stress_Linearization.txt"        # File ( 1st point of arc_length has to be 0! )
H = [0,0,1]                  # Clipping Plane Normal / Hoop Direction

# activate - 1 , deactivate - 0
axisym = 1           # axisymmetric study X - radial direction , Y - axial direction, H = [ 0 , 0 , 1 ]   
print_stress_components = 1
print_von_Mises_stress = 1
print_Tresca_stress = 1
plot_values_in_figures = 1
ignore_bending_in_tangential_direction = 0 # bending stresses are calculated only for the local hoop and meridional (normal) component stresses, according to ASME

tolerance = 1e-5               # tolerance of the plane orthogonality
result_name = "STA_NL__"

# =============================================================================
# STRESS LINEARIZATION - Through Thickness Integration
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from sys import exit

df = pd.read_csv(inp)
# print(df)
collist = ["Points_0","Points_1","Points_2","arc_length",result_name+"SIGM_NOEU_0",result_name+"SIGM_NOEU_1", result_name+"SIGM_NOEU_2", result_name+"SIGM_NOEU_3", result_name+"SIGM_NOEU_4", result_name+"SIGM_NOEU_5" ]
# collist = ["Points_0","Points_1","Points_2","arc_length",result_name+"SIEF_NOEU_0",result_name+"SIEF_NOEU_1", result_name+"SIEF_NOEU_2", result_name+"SIEF_NOEU_3", result_name+"SIEF_NOEU_4", result_name+"SIEF_NOEU_5" ]
df = df.reindex(columns=collist)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
print(df)
data = df.to_numpy()
# print( int(data[0,4]) )

point_1 = [ data[0,0] , data[0,1] , data[0,2] ]
point_2 = [ data[-1,0] , data[-1,1] , data[-1,2] ]

dx = point_2[0]-point_1[0]
dy = point_2[1]-point_1[1]
dz = point_2[2]-point_1[2]
L12 = np.sqrt(dx**2+dy**2+dz**2)

if df["arc_length"][0] != 0:
    print("The first point of arc_length must be 0, but is = " , df["arc_length"][0])
    exit()

if ignore_bending_in_tangential_direction == 1:
    T = [ dx/L12, dy/L12, dz/L12 ]
    N = np.cross(H,T) # ( T[2]*H[1]-T[1]*H[2], T[0]*H[2]-T[2]*H[0], T[1]*H[0]-T[0]*H[1] )
    
    if abs(1-np.linalg.norm(H)) > tolerance:
        print("\033[35m Wrong normal for H,", np.linalg.norm(H), "is not equal to 1")
        exit()
    if abs(np.dot(H,T)) > tolerance:
        print("\033[35m Stress Classification Line (SCL) is not in the clipping plane,", np.dot(H,T), "is not equal to 0" )
        exit()
    print("0 == ", np.dot(H,T))
    print("\n SCL (Stress Classification Line) Orientation:")
    print(" T = Tangential (SCL) --> ", T), print(" N = Normal (Meridional) --> ", N), print(" H = Hoop --> ", H, "magnitude = ", np.linalg.norm(H) )
if ignore_bending_in_tangential_direction == 0:
    T = [ 1,0,0 ] ; N = [ 0,1,0 ] ; H = [ 0,0,1 ] # global system

def global_to_SCL(sx,sy,sz,sxy,sxz,syz,vT,vN,vH):
  
    sigma_SCL_trans = np.zeros((len(sx),6))
    SCL = np.array([ vT,
                     vN,
                     vH ] )
    print("SCL = \n", SCL)
    
    for i in range(0,len(sx)):
        sigma_G = np.array([ [ sx[i], sxy[i], sxz[i] ],
                             [ sxy[i], sy[i], syz[i] ],
                             [ sxz[i], syz[i], sz[i] ] ])
        sigma_SCL = SCL @ sigma_G @ SCL.transpose()
        sigma_SCL_trans[i,:] = np.array([ [sigma_SCL[0,0] , sigma_SCL[1,1] , sigma_SCL[2,2],
                                           sigma_SCL[0,1] , sigma_SCL[0,2] , sigma_SCL[1,2]]  ])
    return sigma_SCL_trans

data_t = data.transpose()
if axisym == 1:
    print(" Axisymmetric Problem")
    stress_trans = global_to_SCL(data_t[4],data_t[5],data_t[6],data_t[7],np.zeros(len(data[:,0])),np.zeros(len(data[:,0])),T,N,H)
else:
    print(" 3D Problem")
    stress_trans = global_to_SCL(data_t[4],data_t[5],data_t[6],data_t[7],data_t[8],data_t[9],T,N,H)
stress_trans_SCL = stress_trans.transpose()
data_T = ( np.concatenate((data[:,:4].transpose(), stress_trans_SCL), axis=0 )).transpose()

x = data_T[:,3]
T = data_T[:,4]
N = data_T[:,5]
H = data_T[:,6]
TN = data_T[:,7]
HT = data_T[:,8]
NH = data_T[:,9]

h = x[-1] ; print(" Line Length = ", h); print(" Number of Points = ", len(x))
if ignore_bending_in_tangential_direction == 1:
    print(" !Bending stresses are calculated only for the local hoop and meridional (normal) component stresses, according to ASME!")

xx = x - h/2

n = 6
mem = np.zeros(n)
memb = np.zeros((n,len(xx)))
for ii in range(0,n):
    # mem[ii] = (1/h) * integrate.simpson( data_T[:,ii+4], xx )
    mem[ii] = (1/h) * np.trapz( data_T[:,ii + 4], xx)
    memb[ii,:] = mem[ii] * np.ones(len(xx))

bend = np.zeros((n,len(xx)))
inte = np.zeros((n,len(xx)))
for ii in range(0,n):
    if ignore_bending_in_tangential_direction == 1:
        if not (ii == 0 or ii == 3 or ii == 4):     # ASME
            inte[ii,:] = data_T[:,ii + 4] * xx
            # bend[ii,:] = xx * (12/h**3) * integrate.simpson( inte[ii,:] , xx)
            bend[ii,:] = xx * (12/h**3) * np.trapz( inte[ii,:] , xx)
    else:
        inte[ii,:] = data_T[:,ii + 4] * xx
        # bend[ii,:] = xx * (12/h**3) * integrate.simpson( inte[ii,:] , xx)
        bend[ii,:] = xx * (12/h**3) * np.trapz( inte[ii,:] , xx)            

real = np.zeros((n,len(xx)))
for ii in range(0,n):
    real[ii,:] = data_T[:,ii + 4]

memb_bend = memb + bend
peak = real - memb_bend

def Tresca_stress(s11,s22,s33,s12,s13,s23):
    tresca_stress = np.zeros(len(s11))
    sigma_1 = np.zeros(len(s11)); sigma_2 = np.zeros(len(s11)); sigma_3 = np.zeros(len(s11))
    for i in range(0,len(s11)):
        sigma = np.array([ [s11[i], s12[i], s13[i] ],
                           [s12[i], s22[i], s23[i] ],
                           [s13[i], s23[i], s33[i] ] ])
        v, n = np.linalg.eig(sigma)
        v = np.sort(v)
        tresca_stress[i] = v[2] - v[0]
        sigma_1[i] = v[2]
        sigma_2[i] = v[1]
        sigma_3[i] = v[0]
    return tresca_stress, sigma_1, sigma_2, sigma_3
Tres_memb, sigma_1_memb, sigma_2_memb, sigma_3_memb = Tresca_stress(memb[0],memb[1],memb[2],memb[3],memb[4],memb[5])
Tres_bend, sigma_1_bend, sigma_2_bend, sigma_3_bend = Tresca_stress(bend[0],bend[1],bend[2],bend[3],bend[4],bend[5])
Tres_memb_bend, sigma_1_memb_bend, sigma_2_memb_bend, sigma_3_memb_bend = Tresca_stress(memb_bend[0],memb_bend[1],memb_bend[2],memb_bend[3],memb_bend[4],memb_bend[5])
Tres_peak, sigma_1_peak, sigma_2_peak, sigma_3_peak = Tresca_stress(peak[0],peak[1],peak[2],peak[3],peak[4],peak[5])
Tres, sigma_1, sigma_2, sigma_3 = Tresca_stress(stress_trans_SCL[0],stress_trans_SCL[1],stress_trans_SCL[2],stress_trans_SCL[3],stress_trans_SCL[4],stress_trans_SCL[5])

# def vMis_stress(s11,s22,s33,s12,s13,s23):
#     stress = np.sqrt( 0.5*((s11-s22)**2+(s22-s33)**2+(s33-s11)**2)+3*(s12**2+s23**2+s13**2) )  
#     return stress
# vM_memb = vMis_stress(memb[0],memb[1],memb[2],memb[3],memb[4],memb[5])
# vM_bend = vMis_stress(bend[0],bend[1],bend[2],bend[3],bend[4],bend[5])
# vM_memb_bend = vMis_stress(memb_bend[0],memb_bend[1],memb_bend[2],memb_bend[3],memb_bend[4],memb_bend[5])
# vM_peak = vMis_stress(peak[0],peak[1],peak[2],peak[3],peak[4],peak[5])
# vM_memb_bend_peak = vM_memb_bend + vM_peak
# vMis = vMis_stress(stress_trans_SCL[0],stress_trans_SCL[1],stress_trans_SCL[2],stress_trans_SCL[3],stress_trans_SCL[4],stress_trans_SCL[5])

def vMis_stress(s1,s2,s3):
    stress = np.sqrt( 0.5*((s1-s2)**2+(s2-s3)**2+(s3-s1)**2))
    return stress
vM_memb = vMis_stress(sigma_1_memb,sigma_2_memb,sigma_3_memb)
vM_bend = vMis_stress(sigma_1_bend,sigma_2_bend,sigma_3_bend)
vM_memb_bend = vMis_stress(sigma_1_memb_bend,sigma_2_memb_bend,sigma_3_memb_bend)
vM_peak = vMis_stress(sigma_1_peak,sigma_2_peak,sigma_3_peak)
vMis = vMis_stress(sigma_1,sigma_2,sigma_3)

plt.close("all")

clr_b = "#6600CC"
clr_m = "#0033CC"
clr_mb = "r"
clr_p = "#0F0F0F"
clr_t = "#4D4D4D"

# =============================================================================
# Stress components
# =============================================================================
if print_stress_components == 1:

    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80 )
    fig1.canvas.manager.set_window_title("Linearized Stress Components")
    
    for kk in range(0,6):
        font = 15
        if kk == 0:
            plt.subplot(3, 2, 1)
            plt.title( "Normal T (radial)"  , fontsize= font)
        if kk == 1:
            plt.subplot(3, 2, 3)
            plt.title( "Normal N (axial)"  , fontsize= font)    
        if kk == 2:
            plt.subplot(3, 2, 5)
            plt.title( "Normal H (tangential)"  , fontsize= font)
        if kk == 3:
            plt.subplot(3, 2, 2)
            plt.title( "Shear TN"  , fontsize= font)
        if kk ==4 :
            plt.subplot(3, 2, 4)
            plt.title( "Shear TH"  , fontsize= font)
        if kk == 5:
            plt.subplot(3, 2, 6)
            plt.title( "Shear NH"  , fontsize= font)    
        
        if ignore_bending_in_tangential_direction == 0:
            if kk == 0:
                plt.subplot(3, 2, 1)
                plt.title( "Component X"  , fontsize= font)
            if kk == 1:
                plt.subplot(3, 2, 3)
                plt.title( "Component Y"  , fontsize= font)    
            if kk == 2:
                plt.subplot(3, 2, 5)
                plt.title( "Component Z"  , fontsize= font)
            if kk == 3:
                plt.subplot(3, 2, 2)
                plt.title( "Component XY"  , fontsize= font)
            if kk ==4 :
                plt.subplot(3, 2, 4)
                plt.title( "Component XZ"  , fontsize= font)
            if kk == 5:
                plt.subplot(3, 2, 6)
                plt.title( "Component YZ"  , fontsize= font)  
        
        if plot_values_in_figures == 1:
            plt.text( x[0], memb[kk,0] ,str(" %.1f"%(memb[kk,0])), color= clr_m  , fontsize= font)
            plt.text( x[0], memb_bend[kk,0] ,str(" %.1f"%(memb_bend[kk,0])), color= clr_mb  , fontsize= font)
            plt.text( x[-1], memb_bend[kk,-1] ,str(" %.1f"%(memb_bend[kk,-1])), color= clr_mb  , fontsize= font)
            plt.text( x[0], bend[kk,0] ,str(" %.1f"%(bend[kk,0])), color= clr_b  , fontsize= font)
            plt.text( x[-1], bend[kk,-1] ,str(" %.1f"%(bend[kk,-1])), color= clr_b  , fontsize= font)    
            plt.text( x[0], peak[kk,0] ,str(" %.1f"%(peak[kk,0])), color= clr_p  , fontsize= font)
            plt.text( x[-1], peak[kk,-1] ,str(" %.1f"%(peak[kk,-1])), color= clr_p  , fontsize= font)
            plt.text( x[0], real[kk,0] ,str(" %.1f"%(real[kk,0])), color= clr_t  , fontsize= font)
            plt.text( x[-1], real[kk,-1] ,str(" %.1f"%(real[kk,-1])), color= clr_t  , fontsize= font)
        plt.rc('xtick', labelsize= 10) 
        plt.rc('ytick', labelsize= 10)
        plt.ylabel('$Stress$' + ' $[MPa]$ ', fontsize = font)
        plt.xlabel('$Coordinate$' + ' $[mm]$ ' , fontsize = font)
        plt.grid(linestyle= '--', linewidth= 2)        
        if kk < 5:
            plt.plot(x, bend[kk], clr_b , linestyle='--', linewidth= 3 )
            plt.plot(x, memb[kk], clr_m , linestyle='-', linewidth= 3 )
            plt.plot(x, memb_bend[kk], clr_mb , linewidth= 4 )
            plt.plot(x, peak[kk], clr_p , linestyle=':', linewidth= 4 )
            plt.plot(x, data_T[:,kk + 4], clr_t , linewidth= 3 )
    
    plt.plot(x, bend[kk], clr_b , label= "Bending Stress", linestyle='--', linewidth= 3 )
    plt.plot(x, memb[kk], clr_m , label= "Membrane Stress", linestyle='-', linewidth= 3 )
    plt.plot(x, memb_bend[kk], clr_mb , label= "Membrane + Bending Stress", linewidth= 4 )
    plt.plot(x, peak[kk], clr_p , label= "Peak Stress", linestyle=':', linewidth= 4 )
    plt.plot(x, data_T[:,kk + 4], clr_t , label= "Total Stress", linewidth= 3 )
    if plot_values_in_figures == 1:
        plt.text( x[0], peak[kk,0] ,str(" %.1f"%(peak[kk,0])), color= clr_p  , fontsize= font)
        plt.text( x[-1], peak[kk,-1] ,str(" %.1f"%(peak[kk,-1])), color= clr_p  , fontsize= font)
        plt.text( x[0], real[kk,0] ,str(" %.1f"%(real[kk,0])), color= clr_t  , fontsize= font)
        plt.text( x[-1], real[kk,-1] ,str(" %.1f"%(real[kk,-1])), color= clr_t  , fontsize= font)
    
    fig1.legend(loc='lower center', shadow= True,  ncol=3, fontsize= 16)
    plt.tight_layout()
    # plt.show()

P = "\033[35m"
B = "\033[36m"
G = "\033[33m"
W = "\033[0m"
R = "\033[37m"

# =============================================================================
# Tresca stress
# =============================================================================
if print_Tresca_stress == 1:

    fig2 = plt.figure(num=None, figsize=(12, 8), dpi=80 )
    fig2.canvas.manager.set_window_title("Linearized Stress Intensity")
    plt.axes( facecolor='#D3FDD8')
    plt.title( "Stress Intensity"  , fontsize= 20)
    plt.plot(x, Tres_bend, clr_b , label= "Bending Stress", linestyle='--', linewidth= 3 )
    plt.plot(x, Tres_memb, clr_m , label= "Membrane Stress", linestyle='-', linewidth= 3 )
    plt.plot(x, Tres_memb_bend, clr_mb , label= "Membrane + Bending Stress", linewidth= 4 )
    plt.plot(x, Tres_peak, clr_p , label= "Peak Stress", linestyle=':', linewidth= 4 )
    plt.plot(x, Tres, clr_t , label= "Total Stress", linewidth= 3 )
    plt.plot(x, Tres, 'h',color = clr_t , markersize= 5, linewidth= 3 )
       
    plt.rc('xtick', labelsize= 10) 
    plt.rc('ytick', labelsize= 10) 
    plt.ylabel('$Stress$' + ' $[MPa]$ ', fontsize = 20)
    plt.xlabel('$Coordinate$' + ' $[mm]$ ' , fontsize = 20)
    
    if plot_values_in_figures == 1:    
        plt.text( x[0], Tres_memb[0] ,str(" %.1f"%(Tres_memb[0])), color= clr_m  , fontsize= 20)
        plt.text( x[0], Tres_bend[0] ,str(" %.1f"%(Tres_bend[0])), color= clr_b  , fontsize= 20)
        plt.text( x[-1], Tres_bend[-1] ,str(" %.1f"%(Tres_bend[-1])), color= clr_b  , fontsize= 20)
        plt.text( x[0], Tres_memb_bend[0] ,str(" %.1f"%(Tres_memb_bend[0])), color= clr_mb  , fontsize= 20)
        plt.text( x[-1], Tres_memb_bend[-1] ,str(" %.1f"%(Tres_memb_bend[-1])), color= clr_mb  , fontsize= 20)
        plt.text( x[0], Tres_peak[0] ,str(" %.1f"%(Tres_peak[0])), color= clr_p , fontsize= 20)
        plt.text( x[-1], Tres_peak[-1] ,str(" %.1f"%(Tres_peak[-1])), color= clr_p, fontsize= 20)
        plt.text( x[0], Tres[0] ,str(" %.1f"%(Tres[0])), color= clr_t  , fontsize= 20)  
        plt.text( x[-1], Tres[-1] ,str(" %.1f"%(Tres[-1])), color= clr_t  , fontsize= 20)

    plt.grid(linestyle= '--', linewidth= 1, color="#4D4D4D")
    fig2.legend(loc='upper center', shadow= True,  ncol=3, fontsize= 16)
    # plt.tight_layout()
    # plt.show()

    print(W + " \n ------------------ Stress Intensity ------------------ " )
           
    print(P + "Bending Stress - Point 1 = ", str(" %.1f"%Tres_bend[0]))
    print(P + "Bending Stress - Point 2 = ", str(" %.1f"%Tres_bend[-1]))
    
    print(B + "Membrane Stress = ", str(" %.1f"%Tres_memb[0]))
    
    print(G + "Membrane + Bending Stress - Point 1 = ", str(" %.1f"%Tres_memb_bend[0]))
    print(G + "Membrane + Bending Stress - Point 2 = ", str(" %.1f"%Tres_memb_bend[-1]))
    print(G + "Membrane + Bending Stress - Point Max = ", str(" %.1f"%(max(Tres_memb_bend))))
    
    print(W + "Peak Stress - Point 1 = ", str(" %.1f"%Tres_peak[0]))
    print(W + "Peak Stress - Point 2 = ", str(" %.1f"%Tres_peak[-1]))
    print(W + "Peak Stress - Max = ", str(" %.1f"%(max(Tres_peak))))
    
    print(R + "Total stress - Point 1 = ", str(" %.1f"%Tres[0]))
    print(R + "Total stress - Point 2 = ", str(" %.1f"%Tres[-1]))
    print(R + "Total stress - Max = ", str(" %.1f"%(max(Tres))))

# =============================================================================
# von Mises stress
# =============================================================================
if print_von_Mises_stress == 1:

    fig3 = plt.figure(num=None, figsize=(12, 8), dpi=80 )
    fig3.canvas.manager.set_window_title("Linearized von Mises Stress")
    plt.axes( facecolor='#DBEDFD')
    plt.title( "von Mises Stress"  , fontsize= 20)
    plt.plot(x, vM_bend, clr_b , label= "Bending Stress", linestyle='--', linewidth= 3 )
    plt.plot(x, vM_memb, clr_m , label= "Membrane Stress", linestyle='-', linewidth= 3 )
    plt.plot(x, vM_memb_bend, clr_mb , label= "Membrane + Bending Stress", linewidth= 4 )
    plt.plot(x, vM_peak, clr_p , label= "Peak Stress", linestyle=':', linewidth= 4 )
    plt.plot(x, vMis, clr_t , label= "Total Stress", linewidth= 3 )
    plt.plot(x, vMis, 's',color = clr_t , markersize= 5, linewidth= 3 )
    
    plt.rc('xtick', labelsize= 10)   
    plt.rc('ytick', labelsize= 10) 
    plt.ylabel('$Stress$' + ' $[MPa]$ ', fontsize = 20)
    plt.xlabel('$Coordinate$' + ' $[mm]$ ' , fontsize = 20)
    if plot_values_in_figures == 1:
        plt.text( x[0], vM_memb[0] ,str(" %.1f"%(vM_memb[0])), color= clr_m  , fontsize= 20)
        plt.text( x[0], vM_bend[0] ,str(" %.1f"%(vM_bend[0])), color= clr_b  , fontsize= 20)
        plt.text( x[-1], vM_bend[-1] ,str(" %.1f"%(vM_bend[-1])), color= clr_b  , fontsize= 20)
        plt.text( x[0], vM_memb_bend[0] ,str(" %.1f"%(vM_memb_bend[0])), color= clr_mb  , fontsize= 20, )
        plt.text( x[-1], vM_memb_bend[-1] ,str(" %.1f"%(vM_memb_bend[-1])), color= clr_mb  , fontsize= 20)
        plt.text( x[0], vM_peak[0] ,str(" %.1f"%(vM_peak[0])), color= clr_p, fontsize= 20)
        plt.text( x[-1], vM_peak[-1] ,str(" %.1f"%(vM_peak[-1])), color= clr_p, fontsize= 20)
        plt.text( x[0], vMis[0] ,str(" %.1f"%(vMis[0])), color= clr_t  , fontsize= 20, horizontalalignment = "right")
        plt.text( x[-1], vMis[-1] ,str(" %.1f"%(vMis[-1])), color= clr_t  , fontsize= 20, horizontalalignment = "right")

    plt.grid(linestyle= '--', linewidth= 1, color="#4D4D4D")
    fig3.legend(loc='upper center', shadow= True,  ncol=3, fontsize= 16)
    # plt.show()
    # plt.tight_layout()
    
    
    print(W + " \n ------------------ von Mises Stress ------------------ " )
     
    print(P + "Bending Stress - Point 1 = ", str(" %.1f"%vM_bend[0]))
    print(P + "Bending Stress - Point 2 = ", str(" %.1f"%vM_bend[-1]))
    
    print(B + "Membrane Stress = ", str(" %.1f"%vM_memb[0]))
    
    print(G + "Membrane + Bending Stress - Point 1 = ", str(" %.1f"%vM_memb_bend[0]))
    print(G + "Membrane + Bending Stress - Point 2 = ", str(" %.1f"%vM_memb_bend[-1]))
    print(G + "Membrane + Bending Stress - Point Max = ", str(" %.1f"%(max(vM_memb_bend))))

    print(W + "Peak stress - Point 1 = ", str(" %.1f"%vM_peak[0]))
    print(W + "Peak stress - Point 2 = ", str(" %.1f"%vM_peak[-1]))
    print(W + "Peak stress - Max = ", str(" %.1f"%(max(vM_peak))))
    
    print(R + "Total Stress - Point 1 = ", str(" %.1f"%vMis[0]))
    print(R + "Total Stress - Point 2 = ", str(" %.1f"%vMis[-1]))
    print(R + "Total Stress - Max = ", str(" %.1f"%(max(vMis))))
    
plt.show()

print(W + "\nVMIS TO EXCEL: Membrane Stress   Membrane + Bending Stress   Total Stress")
print(str("%.1f"%vM_memb[0]) +"\t\t"+str("%.1f"%(max(vM_memb_bend)))+"\t\t"+str("%.1f"%(max(vMis))) )
print(str("%.1f"%vM_memb[0]) +"\t\t"+str("%.1f"%(max(vM_memb_bend)))+"\t\t" )

# plt.figure()
# plt.plot(x , data_T[:,0 + 4]*x)
