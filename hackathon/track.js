// ABOUTME: D3.js visualization for Barber Motorsports Park track analysis
// ABOUTME: Renders track surface with boundaries and interactive zoom/pan

// Rotate coordinates using 2D rotation matrix
function rotateCoordinates(x, y, angleDegrees) {
    const angleRadians = (angleDegrees * Math.PI) / 180;
    const cos = Math.cos(angleRadians);
    const sin = Math.sin(angleRadians);
    return {
        x: x * cos - y * sin,
        y: x * sin + y * cos
    };
}

(async function() {
    // Load track boundaries and centerline
    const boundaries = await d3.json('data/processed/track_boundaries.json');
    const centerline = await d3.csv('data/processed/track_centerline.csv', d => ({
        x: +d.x_meters,
        y: +d.y_meters
    }));

    // Load pit lane
    const pitLane = await d3.json('data/processed/pit_lane.json');

    console.log(`Loaded track boundaries:`);
    console.log(`  Inner: ${boundaries.inner.length} points`);
    console.log(`  Outer: ${boundaries.outer.length} points`);
    console.log(`  Centerline: ${centerline.length} points`);
    console.log(`  Pit lane: ${pitLane.centerline.length} points`);

    // Rotate all coordinates around origin (negative = clockwise)
    const rotationAngle = -42;
    console.log(`Rotating track ${Math.abs(rotationAngle)}° clockwise`);

    // Rotate boundary points
    boundaries.inner = boundaries.inner.map(p => rotateCoordinates(p.x, p.y, rotationAngle));
    boundaries.outer = boundaries.outer.map(p => rotateCoordinates(p.x, p.y, rotationAngle));

    // Rotate centerline points
    centerline.forEach((p, i) => {
        const rotated = rotateCoordinates(p.x, p.y, rotationAngle);
        centerline[i].x = rotated.x;
        centerline[i].y = rotated.y;
    });

    // Rotate pit lane centerline
    pitLane.centerline = pitLane.centerline.map(p => {
        const rotated = rotateCoordinates(p.x_meters, p.y_meters, rotationAngle);
        return { x: rotated.x, y: rotated.y };
    });

    // Rotate pit lane boundaries
    if (pitLane.boundaries) {
        pitLane.boundaries.inner = pitLane.boundaries.inner.map(p => {
            const rotated = rotateCoordinates(p.x_meters, p.y_meters, rotationAngle);
            return { x: rotated.x, y: rotated.y };
        });
        pitLane.boundaries.outer = pitLane.boundaries.outer.map(p => {
            const rotated = rotateCoordinates(p.x_meters, p.y_meters, rotationAngle);
            return { x: rotated.x, y: rotated.y };
        });
    }

    // Set up SVG
    const svg = d3.select('#track-svg');
    const width = window.innerWidth;
    const height = window.innerHeight;

    svg.attr('width', width)
       .attr('height', height);

    // Create defs for checkered pattern
    const defs = svg.append('defs');

    // Checkered flag pattern (minimalistic racing start/finish)
    const squareSize = 0.5; // meters per square (larger squares)
    const checkerPattern = defs.append('pattern')
        .attr('id', 'checkerPattern')
        .attr('patternUnits', 'userSpaceOnUse')
        .attr('width', squareSize * 2)
        .attr('height', squareSize * 2);

    // Background matches track color to hide the line
    checkerPattern.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', squareSize * 2).attr('height', squareSize * 2)
        .attr('fill', '#2a2a2a'); // Same as track color

    checkerPattern.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', squareSize).attr('height', squareSize)
        .attr('fill', 'rgba(180, 180, 180, 0.6)'); // Light gray

    checkerPattern.append('rect')
        .attr('x', squareSize).attr('y', squareSize)
        .attr('width', squareSize).attr('height', squareSize)
        .attr('fill', 'rgba(180, 180, 180, 0.6)'); // Light gray

    // Create main group for zoom/pan
    const g = svg.append('g');

    // Calculate bounds from all boundary points
    const allPoints = [...boundaries.inner, ...boundaries.outer];
    const xExtent = d3.extent(allPoints, d => d.x);
    const yExtent = d3.extent(allPoints, d => d.y);

    const trackWidth = xExtent[1] - xExtent[0];
    const trackHeight = yExtent[1] - yExtent[0];

    // Add padding (10%)
    const padding = 0.1;
    const xPadding = trackWidth * padding;
    const yPadding = trackHeight * padding;

    // Create scales
    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height, 0]); // Invert Y axis (SVG coordinates)

    // Create closed polygon path
    // Go around outer edge clockwise, then inner edge counter-clockwise
    const outerClosed = [...boundaries.outer, boundaries.outer[0]];
    const innerReversed = [...boundaries.inner].reverse();
    const innerClosed = [...innerReversed, innerReversed[0]];

    // Combine into single polygon path
    const trackPolygon = [...outerClosed, ...innerClosed];

    // Line generator with curve smoothing for track
    const line = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveCatmullRom.alpha(0.5));

    // Smoother line generator for pit lane (higher alpha = smoother)
    const smoothLine = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveCatmullRom.alpha(0.8));

    // Draw pit lane FIRST (so it appears behind track)
    if (pitLane.boundaries) {
        const pitOuter = pitLane.boundaries.outer;
        const pitInnerReversed = [...pitLane.boundaries.inner].reverse();
        const pitLanePolygon = [...pitOuter, ...pitInnerReversed];

        g.append('path')
            .datum(pitLanePolygon)
            .attr('class', 'pit-lane-surface')
            .attr('d', smoothLine)
            .style('fill', '#2a2a2a')
            .style('stroke', '#444')
            .style('stroke-width', 1);
    }

    // Draw track surface SECOND (so it appears on top)
    g.append('path')
        .datum(trackPolygon)
        .attr('class', 'track-surface')
        .attr('d', line)
        .style('fill', '#2a2a2a')
        .style('stroke', '#444')
        .style('stroke-width', 1);

    // Load corner labels
    const cornerLabels = await d3.json('data/assets/corner_labels.json');
    console.log(`  Corner labels: ${cornerLabels.length} corners`);

    // Rotate corner label positions
    cornerLabels.forEach(corner => {
        const rotated = rotateCoordinates(corner.x_meters, corner.y_meters, rotationAngle);
        corner.x_meters = rotated.x;
        corner.y_meters = rotated.y;
    });

    // Calculate cumulative distances along centerline
    const centerlineWithDistance = [];
    let cumDistance = 0;

    for (let i = 0; i < centerline.length; i++) {
        if (i === 0) {
            centerlineWithDistance.push({ ...centerline[i], distance: 0 });
        } else {
            const prev = centerline[i - 1];
            const curr = centerline[i];
            const dx = curr.x - prev.x;
            const dy = curr.y - prev.y;
            const segmentDist = Math.sqrt(dx * dx + dy * dy);
            cumDistance += segmentDist;
            centerlineWithDistance.push({ ...curr, distance: cumDistance });
        }
    }

    // Direction triangles removed

    // Draw checkered start/finish marker
    // Calculate track direction at start/finish using ±5 points
    const startIdx = 0;
    const prevIdx = Math.max(0, startIdx - 5);
    const nextIdx = Math.min(centerlineWithDistance.length - 1, startIdx + 5);

    const prevPoint = centerlineWithDistance[prevIdx];
    const nextPoint = centerlineWithDistance[nextIdx];
    const centerPoint = centerlineWithDistance[startIdx];

    // Calculate track angle in degrees
    const dx = nextPoint.x - prevPoint.x;
    const dy = nextPoint.y - prevPoint.y;
    const trackAngle = Math.atan2(dy, dx) * 180 / Math.PI;

    // Calculate track width at start/finish
    const innerPt = boundaries.inner[0];
    const outerPt = boundaries.outer[0];
    const startFinishTrackWidth = Math.sqrt(
        Math.pow(outerPt.x - innerPt.x, 2) +
        Math.pow(outerPt.y - innerPt.y, 2)
    );

    // Checkered flag dimensions
    const flagLength = 20.0; // 20m along track
    const flagWidth = startFinishTrackWidth * 0.95; // 95% of track width (5% smaller)

    // Calculate the perpendicular offset to center the flag on the track
    const perpAngle = (trackAngle + 90) * Math.PI / 180;
    const centerOffset = startFinishTrackWidth / 2;
    const flagCenterX = (innerPt.x + outerPt.x) / 2;
    const flagCenterY = (innerPt.y + outerPt.y) / 2;

    // Transform pattern to match data scale
    const patternScale = checkerPattern
        .attr('patternTransform', `scale(${1 / (xScale(1) - xScale(0))}, ${1 / (yScale(0) - yScale(1))})`);

    g.append('rect')
        .attr('class', 'checkered-start-finish')
        .attr('x', xScale(flagCenterX) - xScale(flagLength / 2) + xScale(0))
        .attr('y', yScale(flagCenterY) - (yScale(0) - yScale(flagWidth / 2)))
        .attr('width', xScale(flagLength) - xScale(0))
        .attr('height', yScale(0) - yScale(flagWidth))
        .attr('transform', `rotate(${trackAngle - 7.5}, ${xScale(flagCenterX)}, ${yScale(flagCenterY)})`)
        .style('fill', 'url(#checkerPattern)')
        .style('opacity', 0.9);

    // Draw corner labels
    const cornerLabelGroup = g.append('g').attr('class', 'corner-labels');

    cornerLabels.forEach(corner => {
        const labelGroup = cornerLabelGroup.append('g')
            .attr('transform', `translate(${xScale(corner.x_meters)}, ${yScale(corner.y_meters)})`);

        // Draw circle background
        labelGroup.append('circle')
            .attr('r', 18)
            .style('fill', 'transparent')
            .style('stroke', '#d4a017')
            .style('stroke-width', 2);

        // Draw corner number
        labelGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .style('fill', '#d4a017')
            .style('font-size', '14px')
            .style('font-weight', '600')
            .text(corner.label);
    });

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.5, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Initial zoom to fit
    const scale = 0.81 * Math.min(
        width / (trackWidth * (1 + 2 * padding)),
        height / (trackHeight * (1 + 2 * padding))
    );

    const centerX = (xExtent[0] + xExtent[1]) / 2;
    const centerY = (yExtent[0] + yExtent[1]) / 2;

    const translateX = width / 2 - xScale(centerX) * scale;
    const translateY = height / 2 - yScale(centerY) * scale;

    svg.call(zoom.transform, d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale));

    console.log('Track visualization ready');

    // Load classified telemetry points
    const classifiedPoints = await d3.csv('data/processed/classified_telemetry_points.csv', d => ({
        race: d.race,
        vehicle_number: parseInt(d.vehicle_number),
        lap: parseInt(d.lap),
        zone_id: parseInt(d.zone_id),
        classification: d.classification,
        x: +d.x_meters,
        y: +d.y_meters
    }));

    console.log(`Loaded ${classifiedPoints.length} classified points`);

    // Rotate classified points
    classifiedPoints.forEach(p => {
        const rotated = rotateCoordinates(p.x, p.y, rotationAngle);
        p.x = rotated.x;
        p.y = rotated.y;
    });

    // Classification colors
    const classificationColors = {
        'Optimal': 'rgb(34, 197, 94)',      // green
        'Conservative': 'rgb(251, 191, 36)', // amber
        'Aggressive': 'rgb(239, 68, 68)'     // red
    };

    // Create group for classified points
    const classifiedPointsGroup = g.append('g').attr('class', 'classified-points');

    // Current state
    let currentRace = 'R1';
    let selectedDrivers = new Set();

    // Load drivers from friction envelopes
    const frictionEnvelopes = await d3.json('data/processed/friction_envelopes.json');
    const drivers = Object.keys(frictionEnvelopes).map(d => parseInt(d)).filter(d => d !== 0).sort((a, b) => a - b);

    console.log(`Loaded ${drivers.length} drivers:`, drivers);

    // Load driver best lap times
    const driverBestLaps = await d3.json('data/processed/driver_best_laps.json');

    // Color scale for drivers
    const driverColors = [
        '#ffd700', '#888888', '#999999', '#777777', '#666666',
        '#aaaaaa', '#555555', '#bbbbbb', '#cccccc', '#444444',
        '#8888aa', '#999977', '#777788', '#666699', '#aaaacc',
        '#5555aa', '#bbbb77', '#cccc88', '#444499', '#8888bb'
    ];

    // Populate driver list
    const driverList = d3.select('#driver-list');

    // Add "All Drivers" option first
    driverList.append('div')
        .attr('class', 'driver-item')
        .attr('data-driver', 'all')
        .html(`
            <div class="driver-color" style="background: #ffd700;"></div>
            <span class="driver-label">All Drivers</span>
        `);

    // Add individual drivers
    drivers.forEach((driver, index) => {
        const color = driverColors[index % driverColors.length];
        const lapTime = driverBestLaps[driver.toString()] || '--:--';

        driverList.append('div')
            .attr('class', 'driver-item')
            .attr('data-driver', driver)
            .html(`
                <div class="driver-color" style="background: ${color};"></div>
                <span class="driver-label">#${driver}</span>
                <span class="driver-time">${lapTime}</span>
            `);
    });

    // Render function for classified points
    function renderClassifiedPoints() {
        // Get selected drivers
        selectedDrivers.clear();
        driverList.selectAll('.driver-item.active[data-driver]:not([data-driver="all"])')
            .each(function() {
                selectedDrivers.add(parseInt(d3.select(this).attr('data-driver')));
            });

        // Filter points by race and selected drivers
        const filteredPoints = classifiedPoints.filter(p => {
            const raceMatch = p.race === currentRace;
            const driverMatch = selectedDrivers.size === 0 || selectedDrivers.has(p.vehicle_number);
            return raceMatch && driverMatch;
        });

        console.log(`Rendering ${filteredPoints.length} points (race: ${currentRace}, drivers: ${selectedDrivers.size})`);

        // Data join
        const circles = classifiedPointsGroup.selectAll('circle')
            .data(filteredPoints, d => `${d.race}-${d.vehicle_number}-${d.lap}-${d.zone_id}`);

        // Enter
        circles.enter()
            .append('circle')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .attr('fill', d => classificationColors[d.classification])
            .attr('fill-opacity', 0.7)
            .attr('stroke', d => classificationColors[d.classification])
            .attr('stroke-width', 0.5);

        // Exit
        circles.exit().remove();
    }

    // Set up race selector
    d3.selectAll('.race-btn').on('click', function() {
        // Update active state
        d3.selectAll('.race-btn').classed('active', false);
        d3.select(this).classed('active', true);

        // Update current race
        currentRace = d3.select(this).attr('data-race');
        console.log(`Race changed to: ${currentRace}`);

        // Re-render points
        renderClassifiedPoints();
    });

    // Set up driver selector interactions
    driverList.selectAll('.driver-item').on('click', function() {
        const driverValue = d3.select(this).attr('data-driver');

        if (driverValue === 'all') {
            // Toggle all drivers
            const allDriverItems = driverList.selectAll('.driver-item[data-driver]:not([data-driver="all"])');
            const anyActive = allDriverItems.filter('.active').size() > 0;

            if (anyActive) {
                // Unmark all
                allDriverItems.classed('active', false);
                d3.select(this).classed('active', false);
                console.log('All drivers unmarked');
            } else {
                // Mark all
                allDriverItems.classed('active', true);
                d3.select(this).classed('active', true);
                console.log('All drivers marked');
            }
        } else {
            // Individual driver toggle
            const isActive = d3.select(this).classed('active');
            d3.select(this).classed('active', !isActive);

            // Update "All Drivers" state
            const allDriverItems = driverList.selectAll('.driver-item[data-driver]:not([data-driver="all"])');
            const allActive = allDriverItems.filter('.active').size() === allDriverItems.size();
            driverList.select('.driver-item[data-driver="all"]').classed('active', allActive);

            console.log(`Driver #${driverValue}: ${!isActive ? 'selected' : 'deselected'}`);
        }

        // Re-render points
        renderClassifiedPoints();
    });

    // Set up control button interactions
    d3.selectAll('.control-btn').on('click', function() {
        const isActive = d3.select(this).classed('active');
        d3.select(this).classed('active', !isActive);

        const buttonId = d3.select(this).attr('id');
        const buttonText = d3.select(this).text();
        console.log(`${buttonText}: ${!isActive ? 'ON' : 'OFF'}`);

        // Handle corner labels visibility
        if (buttonId === 'toggle-corners') {
            cornerLabelGroup.style('display', !isActive ? 'block' : 'none');
        }

        // Handle average events visibility
        if (buttonId === 'toggle-avg-events') {
            classifiedPointsGroup.style('display', !isActive ? 'block' : 'none');
        }
    });

    // Initial render
    renderClassifiedPoints();
    console.log('Classified points visualization ready');

    // ========================================
    // Zone Navigation System
    // ========================================

    // Load turn zones and calculate bounding boxes
    const turnZones = await d3.json('data/processed/turn_zones.json');
    console.log(`Loaded ${turnZones.length} turn zones`);

    // Calculate bounding box for each zone from classified points
    const ZONE_DATA = [];

    // Add "Full Track" view first
    ZONE_DATA.push({
        label: 'Full Track',
        xRange: [xExtent[0] - xPadding, xExtent[1] + xPadding],
        yRange: [yExtent[0] - yPadding, yExtent[1] + yPadding]
    });

    // Calculate bounds for each turn zone with minimum comfortable size
    const MINIMUM_ZONE_SIZE = 150; // meters - ensures enough context

    turnZones.forEach(zone => {
        const zonePoints = classifiedPoints.filter(p => p.zone_id === zone.zone_id);

        if (zonePoints.length > 0) {
            const xExtentZone = d3.extent(zonePoints, p => p.x);
            const yExtentZone = d3.extent(zonePoints, p => p.y);

            // Calculate natural dimensions
            let zoneWidth = xExtentZone[1] - xExtentZone[0];
            let zoneHeight = yExtentZone[1] - yExtentZone[0];

            // Apply minimum size constraint
            const effectiveWidth = Math.max(zoneWidth, MINIMUM_ZONE_SIZE);
            const effectiveHeight = Math.max(zoneHeight, MINIMUM_ZONE_SIZE);

            // Center the box around the original center
            const centerX = (xExtentZone[0] + xExtentZone[1]) / 2;
            const centerY = (yExtentZone[0] + yExtentZone[1]) / 2;

            const xMin = centerX - effectiveWidth / 2;
            const xMax = centerX + effectiveWidth / 2;
            const yMin = centerY - effectiveHeight / 2;
            const yMax = centerY + effectiveHeight / 2;

            // Add padding (20% of effective size)
            const zonePadding = 0.2;
            const xPaddingZone = effectiveWidth * zonePadding;
            const yPaddingZone = effectiveHeight * zonePadding;

            ZONE_DATA.push({
                label: zone.name,
                xRange: [xMin - xPaddingZone, xMax + xPaddingZone],
                yRange: [yMin - yPaddingZone, yMax + yPaddingZone]
            });
        }
    });

    console.log(`Zone navigation ready with ${ZONE_DATA.length} zones`);

    // State management
    let activeZoneIndex = 0;
    let isAnimating = false;

    // Smooth zoom animation using D3 transforms
    function smoothZoom(targetXRange, targetYRange, duration, callback) {
        const currentTransform = d3.zoomTransform(svg.node());

        // Calculate target transform from bounding box
        const targetScale = 0.95 * Math.min(
            width / (xScale(targetXRange[1]) - xScale(targetXRange[0])),
            height / (yScale(targetYRange[0]) - yScale(targetYRange[1]))
        );

        const targetCenterX = (targetXRange[0] + targetXRange[1]) / 2;
        const targetCenterY = (targetYRange[0] + targetYRange[1]) / 2;

        const targetTranslateX = width / 2 - xScale(targetCenterX) * targetScale;
        const targetTranslateY = height / 2 - yScale(targetCenterY) * targetScale;

        const targetTransform = d3.zoomIdentity
            .translate(targetTranslateX, targetTranslateY)
            .scale(targetScale);

        // Animation loop
        const startTime = performance.now();

        function step(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1.0);

            // Cubic ease-in-out
            const eased = progress < 0.5
                ? 4 * progress * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            // Interpolate transform
            const k = currentTransform.k + (targetTransform.k - currentTransform.k) * eased;
            const x = currentTransform.x + (targetTransform.x - currentTransform.x) * eased;
            const y = currentTransform.y + (targetTransform.y - currentTransform.y) * eased;

            const interpolatedTransform = d3.zoomIdentity
                .translate(x, y)
                .scale(k);

            svg.call(zoom.transform, interpolatedTransform);

            if (progress < 1.0) {
                requestAnimationFrame(step);
            } else {
                if (callback) callback();
            }
        }

        requestAnimationFrame(step);
    }

    // Navigate to specific zone
    function navigateToZone(zoneIndex) {
        if (isAnimating || zoneIndex === activeZoneIndex) return;

        const zoneData = ZONE_DATA[zoneIndex];
        if (!zoneData) return;

        console.log(`Navigating to: ${zoneData.label}`);

        isAnimating = true;
        smoothZoom(zoneData.xRange, zoneData.yRange, 500, function() {
            activeZoneIndex = zoneIndex;
            isAnimating = false;
        });
    }

    // Keyboard event listener (left/right arrows only)
    document.addEventListener('keydown', function(e) {
        // Ignore if typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        // Prevent animation queueing
        if (isAnimating) {
            return;
        }

        const maxIndex = ZONE_DATA.length - 1;
        let newIndex = activeZoneIndex;

        // Left/right arrow navigation with wrap-around
        if (e.key === 'ArrowRight') {
            newIndex = (activeZoneIndex + 1) % (maxIndex + 1);
            e.preventDefault();
        } else if (e.key === 'ArrowLeft') {
            newIndex = (activeZoneIndex - 1 + maxIndex + 1) % (maxIndex + 1);
            e.preventDefault();
        }

        if (newIndex !== activeZoneIndex) {
            navigateToZone(newIndex);
        }
    });

    console.log('Keyboard navigation active (← → to cycle through zones)');
})();
