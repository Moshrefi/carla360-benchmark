"""Publication-exportable figures for Standard_GT statistics."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def make_figures(result):
    xyz = result['xyz']; local = xyz-xyz[0]
    frame = np.arange(len(xyz)); pair = frame[1:]
    cm = 100*result['step_distance_m']; deg = result['rotation_step_deg']
    gap = result['source_id_delta'] > 1
    figures = {}
    with plt.rc_context({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2}):
        fig = plt.figure(figsize=(14,6), layout='constrained')
        fig.get_layout_engine().set(wspace=.22)
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(*local.T, color='#94a3b8', lw=.8)
        points = ax.scatter(*local.T, c=frame, cmap='viridis', s=8)
        ax.scatter(*local[0], color='green', marker='o', s=70, label='Start')
        ax.scatter(*local[-1], color='red', marker='X', s=70, label='End')
        ax.set(xlabel='X offset (m)', ylabel='Y offset (m)', zlabel='Z offset (m)', title='3D camera trajectory')
        extent = np.ptp(local, axis=0)
        centre = (local.min(axis=0)+local.max(axis=0))/2
        radius = max(float(extent.max())*.55, .001)
        for j, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(centre[j]-radius, centre[j]+radius)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_major_locator(MaxNLocator(4))
        ax.set_box_aspect((1,1,1))  # Equal physical scale, also for planar/stationary paths.
        ax.legend()
        ax = fig.add_subplot(122)
        ax.plot(local[:,0],local[:,1], color='#94a3b8',lw=.8)
        ax.scatter(local[:,0],local[:,1],c=frame,cmap='viridis',s=10)
        ax.scatter(*local[0,:2], color='green',s=70,label='Start')
        ax.scatter(*local[-1,:2], color='red',marker='X',s=70,label='End')
        ax.set(xlabel='X offset (m)',ylabel='Y offset (m)',title='Top view; coordinates relative to first frame')
        ax.set_aspect('equal',adjustable='datalim'); ax.legend()
        fig.colorbar(points,ax=fig.axes,label='Retained frame index',shrink=.75)
        figures['trajectory'] = fig

        fig, axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for j,name in enumerate(['X','Y','Z']):
            axes[0,0].plot(frame,local[:,j],label=name)
        axes[0,0].set(ylabel='Offset from first frame (m)',title='Position components'); axes[0,0].legend()
        axes[0,1].plot(frame,xyz[:,2]); axes[0,1].set(ylabel='World Z (m)',title='World Z; not height above terrain')
        axes[1,0].plot(frame,result['cumulative_distance_m'])
        axes[1,0].set(ylabel='Distance (m)',title='Cumulative retained-frame path length')
        axes[1,1].plot(frame,result['rotation_from_first_deg'])
        axes[1,1].set(ylabel='SO(3) angle (deg)',title='Orientation difference from first frame',ylim=(0,180))
        for ax in axes.flat: ax.set_xlabel('Retained frame index')
        figures['position_and_reference_rotation'] = fig

        fig, axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for row, (values,label) in enumerate([(cm,'Translation (cm)'),(deg,'Rotation (deg)')]):
            ax=axes[row,0]; ax.plot(pair,values,lw=.8,color='#2563eb')
            if gap.any(): ax.scatter(pair[gap],values[gap],s=15,color='#ea580c',label='Source-ID gap'); ax.legend()
            ax.set(xlabel='Destination retained frame',ylabel=label,title='Adjacent retained frames')
            ax=axes[row,1]
            if len(values):
                ax.hist(values,bins=min(50,max(1,int(np.sqrt(len(values))))),color='#2563eb',alpha=.8)
                ax.axvline(np.median(values),color='#dc2626',label='Median');ax.legend()
            else: ax.text(.5,.5,'No adjacent pairs',ha='center',transform=ax.transAxes)
            ax.set(xlabel=label,ylabel='Pair count',title='Distribution')
        figures['adjacent_motion'] = fig

        fig, axes = plt.subplots(3,2,figsize=(12,9),layout='constrained')
        for j,name in enumerate(['Pitch','Yaw','Roll']):
            values=result['euler_deg'][:,j]
            axes[j,0].scatter(frame,values,s=3)
            axes[j,0].set(xlabel='Retained frame index',ylabel=name+' (deg)',title='Original CARLA '+name.lower())
            axes[j,1].hist(values,bins=min(50,max(1,int(np.sqrt(len(values))))),color='#0d9488')
            axes[j,1].set(xlabel=name+' (deg)',ylabel='Frame count',title='Source Euler distribution')
        figures['source_euler_angles'] = fig
    return figures


def interactive_trajectory(result):
    """Optional Plotly view, using equal scale for metric axes."""
    import plotly.graph_objects as go
    xyz = result['xyz']-result['xyz'][0]
    ids = result['source_ids']
    fig = go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines+markers',
        line={'width':2,'color':'#94a3b8'},
        marker={'size':3,'color':list(range(len(ids))),'colorscale':'Viridis','showscale':True,
                'colorbar':{'title':'Frame'}},
        customdata=np.column_stack([np.arange(len(ids)),ids]),
        hovertemplate='Frame %{customdata[0]} | Source ID %{customdata[1]}<br>X %{x:.3f} m<br>Y %{y:.3f} m<br>Z %{z:.3f} m<extra></extra>'))
    fig.update_layout(title='Trajectory relative to first camera centre',height=650,
        scene={'xaxis_title':'X offset (m)','yaxis_title':'Y offset (m)',
               'zaxis_title':'Z offset (m)','aspectmode':'data'})
    return fig
