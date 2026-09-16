"""Native-camera presentation only; has no flight-control interface."""
import math
import cv2
import numpy as np

CAMERA_POSITION = np.array([0., -9., 6.])
CAMERA_PITCH = .45
CAMERA_FOV = 1.20
COLORS = [(40, 165, 245), (245, 110, 40), (175, 90, 235)]  # RGB


def project(point, width, height):
    delta = np.asarray(point) - CAMERA_POSITION
    # Gazebo camera optical axis is model +X; yaw=pi/2, pitch=.45.
    depth = math.cos(CAMERA_PITCH)*delta[1] - math.sin(CAMERA_PITCH)*delta[2]
    vertical = math.sin(CAMERA_PITCH)*delta[1] + math.cos(CAMERA_PITCH)*delta[2]
    if depth <= .1:
        return None
    focal = width/(2*math.tan(CAMERA_FOV/2))
    return np.array([width/2 + focal*delta[0]/depth, height/2-focal*vertical/depth])


def payload_reference(t, config):
    h, d = config['hold_duration'], config['leg_duration']
    def law(u):
        return 35*u**4-84*u**5+70*u**6-20*u**7
    s = 0.
    if h <= t < h+d: s = law((t-h)/d)
    elif h+d <= t < 2*h+d: s = 1.
    elif 2*h+d <= t < 2*h+2*d: s = 1-law((t-2*h-d)/d)
    return np.array([-3+6*s, math.sin(2*math.pi*s), config['payload_com_height']])


class Presentation:
    def __init__(self, config):
        self.config = config
        self.history = []
        self.duration=2*config['leg_duration']+3*config['hold_duration']
        self.hold=config['hold_duration'];self.leg=config['leg_duration']

    def render(self, rgb, stamp, stage, transport_start, truth, cables, states, truth_stamp=None):
        native = rgb.copy(); h, w = native.shape[:2]
        payload = None
        for name, pose in zip(truth.name, truth.pose) if truth is not None else []:
            p = pose.position
            if name == 'payload': payload = pose
            if name in ('uav1', 'uav2', 'uav3'):
                point = project([p.x, p.y, p.z], w, h)
                if point is not None:
                    pos = tuple((point+[10, -14]).astype(int)); i = int(name[-1])-1
                    cv2.putText(native, 'UAV %d'%(i+1), pos, cv2.FONT_HERSHEY_SIMPLEX, .55, (15,20,25), 4, cv2.LINE_AA)
                    cv2.putText(native, 'UAV %d'%(i+1), pos, cv2.FONT_HERSHEY_SIMPLEX, .55, COLORS[i], 1, cv2.LINE_AA)
        # Entire original view fits the scene viewport. No close-up or crop.
        frame = np.full((h,w,3), (12,22,32), dtype=np.uint8)
        frame[64:604, :960] = cv2.resize(native, (960,540), interpolation=cv2.INTER_AREA)
        def text(s, pos, scale=.48, color=(235,242,248)):
            cv2.putText(frame,s,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        elapsed = None if transport_start is None else stamp-transport_start
        phase = stage.upper().replace('_',' ')
        if stage == 'transport' and elapsed is not None:
            phase = ('START HOLD' if elapsed < self.hold else 'OUTBOUND A > B' if elapsed < self.hold+self.leg else 'TURNAROUND / HOLD' if elapsed < 2*self.hold+self.leg else 'RETURN B > A' if elapsed < 2*self.hold+2*self.leg else 'FINISH HOLD')
        text('THREE-UAV COOPERATIVE TRANSPORT', (20,27),.65)
        text('Gazebo / ODE + ROS + PX4 SITL | SIMULATION', (20,49),.43,(170,190,205))
        text(phase,(650,27),.55)
        text('SIM %.1f s | TRANSPORT %s'%(stamp,'--' if elapsed is None else '%.1f / %.1f s'%(np.clip(elapsed,0,self.duration),self.duration)),(650,49),.43)
        tension = None; error = None; tilt = None; signed = None
        if cables is not None and 0 <= stamp-cables[0] <= .25:
            tension = np.array([cables[1+3*i] for i in range(3)])
        if payload is not None and truth_stamp is not None and 0 <= stamp-truth_stamp <= .25:
            q=payload.orientation
            norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
            if norm > 0: tilt=math.degrees(math.acos(np.clip(1-2*(q.x*q.x+q.y*q.y)/norm,-1,1)))
            if stage == 'transport' and transport_start is not None:
                p=payload.position
                signed=(np.array([p.x,p.y,p.z])-payload_reference(truth_stamp-transport_start,self.config))*100
                error=float(np.linalg.norm(signed))
        if stage == 'transport' and elapsed is not None and 0 <= elapsed <= self.duration:
            self.history.append((elapsed, tension.copy() if tension is not None else np.full(3,np.nan),error if error is not None else np.nan,tilt if tilt is not None else np.nan))
        text('SYNCHRONIZED TELEMETRY',(977,85),.43)
        text('Gazebo truth | transport only',(977,104),.36,(170,190,205))
        def plot(top,title,values,colors,limits,unit):
            x0,x1,y0,y1=1015,1260,top+39,top+119
            text(title,(977,top+18),.43)
            low,high=limits
            for level in np.linspace(low,high,3):
                y=int(y1-(level-low)/(high-low)*(y1-y0))
                cv2.line(frame,(x0,y),(x1,y),(40,55,65),1)
                text('%.1f'%level,(978,y+4),.32,(170,190,205))
            for t in [0,self.duration/2,self.duration]:
                x=int(x0+t/self.duration*(x1-x0));cv2.line(frame,(x,y0),(x,y1),(40,55,65),1)
                text('%g'%t,(x-7,y1+15),.30,(170,190,205))
            text(unit,(978,y0-7),.31,(170,190,205));text('transport / s',(1150,y1+30),.31,(170,190,205))
            if self.history:
                clock=np.array([row[0] for row in self.history]);data=np.asarray(values)
                if data.ndim==1:data=data[:,None]
                for i,color in enumerate(colors):
                    valid=np.isfinite(data[:,i]);indices=np.flatnonzero(valid)
                    for run in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                        if len(run)<2:continue
                        px=x0+clock[run]/self.duration*(x1-x0);py=y1-(np.clip(data[run,i],low,high)-low)/(high-low)*(y1-y0)
                        cv2.polylines(frame,[np.column_stack((px,py)).astype(np.int32)],False,color,1,cv2.LINE_AA)
        ts=[row[1] for row in self.history];es=[row[2] for row in self.history];angles=[row[3] for row in self.history]
        finite_t=[v for row in ts for v in row if np.isfinite(v)]
        tension_limits=(min(9.5,math.floor(min(finite_t+[9.5])*2)/2),max(11.5,math.ceil(max(finite_t+[11.5])*2)/2))
        plot(115,'ROPE TENSION | U1 U2 U3',ts,COLORS,tension_limits,'N')
        plot(280,'PAYLOAD POSITION ERROR',es,[(85,215,180)],(0,max(4.,math.ceil(max([v for v in es if np.isfinite(v)]+[0])/2)*2)),'cm')
        plot(445,'PAYLOAD TILT',angles,[(240,200,90)],(0,max(1.,math.ceil(max([v for v in angles if np.isfinite(v)]+[0])))),'deg')
        for i in range(3):
            state=states[i]
            text('UAV %d | %s | %s'%(i+1,state.mode if state else 'WAIT','ARMED' if state and state.armed else 'DISARMED'),(20+i*420,635),.43,COLORS[i])
            text('T%d: %s N'%(i+1,'--' if tension is None else '%.3f'%tension[i]),(20+i*420,660),.58,COLORS[i])
        text('LOAD ERROR: %s cm'%('--' if error is None else '%.2f'%error),(20,690),.55,(85,215,180))
        text('XYZ: %s cm'%('--' if signed is None else ' / '.join('%+.2f'%v for v in signed)),(355,690),.48)
        text('LOAD TILT: %s deg'%('--' if tilt is None else '%.3f'%tilt),(890,690),.48,(240,200,90))
        text('Error = ||load position - reference|| | no error reference during takeoff / landing',(20,710),.32,(170,190,205))
        if elapsed is not None:
            cv2.rectangle(frame,(0,h-3),(int(w*np.clip(elapsed/self.duration,0,1)),h-1),(60,180,235),-1)
        return frame
