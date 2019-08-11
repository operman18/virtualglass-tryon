function compute_opacity(tryon){
    // We try to not show very large angle where prediction is unreliabe
    // Moreover, getting rid
    var a = Math.abs(tryon.rotation.x) + Math.abs(tryon.rotation.y)*2.5 + Math.abs(tryon.rotation.z) + Math.abs(tryon.rotation.y)*1.5*Math.abs(tryon.rotation.z)*4;
    var lag = Math.min(Math.abs(new Date()-tryon.begin),10000);
    var p = (Math.pow(0.99,lag));
    p = Math.min(p,0.95);
    opacity = p*tryon.opacity + (1-p)*tryon.data['state']/0.4*((a<0.6)?0.6:((a>1.1)?0.0:1.3-a));
    if( opacity)
    {
	return opacity;
    }
    else
    {
	return 0;
    }
}

var TryOnFace = function (params) {
    var ref = this;
    
    this.selector = params.selector;
    this.selector_real = params.live_selector;

    //sizes
    this.fps = params.fps;
    this.speed = params.speed;
    this.momentum = params.momentum;
    this.momentum_opacity = params.momentum_opacity;
    this.opacity = params.opacity;
    this.object = params.object;
    this.width = params.width;
    this.height = params.height;
	   
    if (params.statusHandler) {
	this.statusHandler = params.statusHandler;
    } else {
	this.statusHandler = function(){};
    }

    this.changeStatus = function(status) {
	this.status = status;
	this.statusHandler(this.status);
    };

    this.changeStatus('STATUS_READY');
    
    if (params.debug) {
	this.debug = true;
	this.debugMsg = this.status;
    } else {
	this.debug = false;
    }
    
    /* CAMERA */
    this.video = document.getElementById('camera');
    
    document.getElementById(this.selector).style.width = this.width + "px";
    document.getElementById(this.selector_real).style.width = this.width + "px";
    this.video.setAttribute('width', this.width);
    this.video.setAttribute('height', this.height);
    	   
    this.debug = function(msg) {
	if (this.debug) {
	    this.debugMsg += msg + "<br>";
	}
    };
    
    this.start = function() {
	var vid = document.getElementById('camera-fake');
	
	var errorElement = document.querySelector('#errorMsg');
	
	// Put variables in global scope to make them available to the browser console.
	var constraints = window.constraints = {
	    audio: false,
	    video: true
	};
	
	function handleSuccess(stream) {
	    vid.srcObject = stream;
	    vid.play();
	}
	
	function handleError(error) {
	    ref.changeStatus('STATUS_CAMERA_ERROR');
	}
	
	navigator.mediaDevices.getUserMedia(constraints).
	    then(handleSuccess).catch(handleError);
	
	
	//continue in loop
	ref.rotation.x = 0;
    };
    
    
    this.printDebug = function() {
	if (this.debug) {
	    document.getElementById('debug').innerHTML = 'the message '+this.debugMsg;
	    this.debugMsg = '';
	}
    };

    this.update = function(stream){
	ref.data = stream.data;
	ref.fps = stream.fps;
	ref.speed = stream.speed;
	ref.begin = stream.begin;
	ref.momentum = stream.momentum;
    }
    
    this.loop = function() {
	requestAnimFrame(ref.loop);
	if(ref.data)
	{
	    ref.position.x = ref.position.x*(1-ref.momentum)+ref.momentum*ref.data.position.x;
	    ref.position.y = ref.position.y*(1-ref.momentum)+ref.momentum*ref.data.position.y;
	    ref.position.z = ref.position.z*(1-ref.momentum)+ref.momentum*ref.data.position.z;
            ref.size.x = ref.data.size.x;
	    ref.rotation.z = (1-ref.momentum)*ref.rotation.z+ref.momentum*ref.data.rotation.z;
	    ref.rotation.y = (1-ref.momentum)*ref.rotation.y+ref.momentum*ref.data.rotation.y;
	    ref.rotation.x = (1-ref.momentum)*ref.rotation.x+ref.momentum*ref.data.rotation.x;
	}
	else
	{
	    ref.changeStatus('STATUS_SEARCH');
	}
	ref.debug(ref.status);
	       
	//print debug
	ref.printDebug();
	ref.render();
	
    };
    
    /* 3D */
    var canvas = document.getElementById("overlay");
    var renderer = new THREE.WebGLRenderer({
	canvas: canvas,
	antialias: true,
	alpha: true
    });
    renderer.setClearColor(0xffffff, 0);
    renderer.setSize(this.width, this.height);
    
    
    //add scene
    var scene = new THREE.Scene;
    
    //define sides
    var outside = {
	1 : 'left',
	0 : 'right',
	//2 : 'front',
	//3 : 'front',
	//4 : 'front',
	5 : 'front'
    };
        
    this.images = [];
    var materials = [];
    for (i = 0; i < 6; i++) {
	if (this.object.outside[outside[i]] !== undefined) {
	    var image = new Image();
	    image.src = this.object.outside[outside[i]];
	    this.images[outside[i]] = image;
	    materials.push(new THREE.MeshLambertMaterial({
		map: THREE.ImageUtils.loadTexture(this.object.outside[outside[i]]), transparent: true
	    }));
	} else {
	    materials.push(new THREE.MeshLambertMaterial({
		color: 0xffffff, transparent: true, opacity: 0
	    }));
	}
    }
    
    //init position and size
    this.position = {
	x: 0,
	y: 0,
	z: 1
    };
    this.rotation = {
	x: 0,
	y: 0,
	z: 0
    };
    this.size = {
	x: 1,
	y: 1,
	z: 1
    };
    
    //set up object
    var geometry = new THREE.CubeGeometry(1, 1, 1);
    var materials = new THREE.MeshFaceMaterial(materials);
    var cube = new THREE.Mesh( geometry, materials );
    cube.doubleSided = true;
    scene.add(cube);
    
    //set up camera
    /*
    var fov = 89.82;
    var camera = new THREE.PerspectiveCamera(fov, this.width / this.height, 1, 5000);
    camera.position.x = -this.width/2 ;
    camera.position.y = this.height/2 ;
    camera.lookAt(this.position);
    */
    var lvf = this.width/-2;
    var rvf = this.width/2;
    var tvf = this.height/2;
    var bvf = this.height/-2;
    var nvf = -1000;
    var fvf = 5000;    
    var camera = new THREE.OrthographicCamera(lvf,
					      rvf,
					      tvf,
					      bvf,
					      nvf,
					      fvf);
    //camera.position.x = -this.width/2 ;
    //camera.position.y = this.height/2 ;
    camera.lookAt(this.position);
    scene.add(camera);

    this.camera_tmp = camera ;
    
    //set up lights
    var lightFront = new THREE.PointLight(0xffffff);
    lightFront.position.set(0, 0, -1000);
    lightFront.intensity = 1.5;
    scene.add(lightFront);
    
    var lightLeft = new THREE.PointLight(0xffffff);
    lightLeft.position.set(1000, 0, 0);
    lightLeft.intensity = 0.7;
    scene.add(lightLeft);
    
    var lightRight = new THREE.PointLight(0xffffff);
    lightRight.position.set(-1000, 0, 0);
    lightRight.intensity = 0.7;
    scene.add(lightRight);
    
    // Define euler angle order for pose
    this.order = 'XYZ';
    
    this.render = function() {
	if(ref.data)
	{
	    //update position
	    cube.position.x = this.position.x;
	    cube.position.y = this.position.y;
	    cube.position.z = this.position.z;
	    
	    cube.rotation.x = this.rotation.x;
	    cube.rotation.y = this.rotation.y;
	    cube.rotation.z = this.rotation.z;
	    
	    // compute opacity
	    if(ref.opacity)
	    {
		opacity = compute_opacity(ref)
		opacity_w = ref.opacity*(1-ref.momentum_opacity)+ref.momentum_opacity*opacity;
		ref.opacity = opacity_w*0.99;
	    }
	    if(!ref.opacity)
	    {
		console.log("this.opacity is undefined")
		ref.opacity = 0.5;
	    }


	    //We add momentum to the new opacity

	    cube.material.materials[0].opacity = ref.opacity;
	    cube.material.materials[1].opacity = ref.opacity;
	    cube.material.materials[5].opacity = ref.opacity;
	    
	    // upate size
	    cube.scale.x = this.size.x;
	    cube.scale.y = this.size.x/800*250;
	    cube.scale.z = this.size.x;
	    cube.eulerOrder = this.order
	    
	    // Slow down the script to avoid quick update
	    if(Math.abs(new Date()-ref.begin)*ref.fps>600*ref.speed)
	    { 
		renderer.render(scene, camera);
	    }
	}
    };
    
    //print debug
    this.printDebug();
};
