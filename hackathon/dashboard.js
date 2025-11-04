// ABOUTME: D3.js visualization for Barber Motorsports Park track analysis
// ABOUTME: Renders track surface with boundaries and interactive zoom/pan

(async function() {
    // Load track boundaries and centerline
    const boundaries = await d3.json('data/processed/track_boundaries.json');
    const centerline = await d3.csv('data/processed/track_centerline.csv', d => ({
        x: +d.x_meters,
        y: +d.y_meters
    }));

    console.log(`Loaded track boundaries:`);
    console.log(`  Inner: ${boundaries.inner.length} points`);
    console.log(`  Outer: ${boundaries.outer.length} points`);
    console.log(`  Centerline: ${centerline.length} points`);

    // Set up SVG
    const svg = d3.select('#track-svg');
    const width = window.innerWidth;
    const height = window.innerHeight;

    svg.attr('width', width)
       .attr('height', height);

    // Create separate groups for track (rotated) and UI elements (not rotated)
    const trackGroup = svg.append('g').attr('class', 'track-elements');
    const uiGroup = svg.append('g').attr('class', 'ui-elements');

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

    // Line generator
    const line = d3.line()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveLinear);

    // Draw track surface as filled polygon
    trackGroup.append('path')
        .datum(trackPolygon)
        .attr('class', 'track-surface')
        .attr('d', line)
        .style('fill', '#2a2a2a')
        .style('stroke', '#444')
        .style('stroke-width', 1);

    // Load corner labels
    const cornerLabels = await d3.json('data/assets/corner_labels.json');
    console.log(`  Corner labels: ${cornerLabels.length} corners`);

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

    // Draw corner labels in UI group (not rotated)
    const cornerLabelGroup = uiGroup.append('g').attr('class', 'corner-labels');

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

    // Rotation angle (positive for clockwise in SVG)
    const rotationAngle = 40;

    // Add zoom behavior with rotation on track only
    const zoom = d3.zoom()
        .scaleExtent([0.5, 10])
        .on('zoom', (event) => {
            // Apply zoom transform with rotation to track, without rotation to UI
            const cx = width / 2;
            const cy = height / 2;

            // Track group: rotated + zoom/pan
            trackGroup.attr('transform',
                `translate(${cx},${cy}) rotate(${rotationAngle}) translate(${-cx},${-cy}) ${event.transform}`
            );

            // UI group: zoom/pan only (no rotation)
            uiGroup.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Initial zoom to fit
    const scale = 0.9 * Math.min(
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
    });
})();
