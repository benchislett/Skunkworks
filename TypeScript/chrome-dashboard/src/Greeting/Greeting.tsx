import React, { useEffect } from 'react';

import moment from 'moment';

import './Greeting.css';
import { Typography } from '@material-ui/core';

const getTime = () => moment().format('hh:mm A');

function Clock() {
  const [time, setTime] = React.useState(getTime);

  useEffect(() => {
    const interval = setInterval(() => setTime(getTime()), 20000);
    return () => clearInterval(interval);
  }, []);

  return <Typography variant='h2'>{time}</Typography>;
}

function Greeting() {
  return (
    <div className='greeting-root'>
      <div className='greeting-time'>
        <Clock />
      </div>
      <div className='greeting-hello'>
        <Typography>Good Morning, Benjamin</Typography>
      </div>
    </div>
  );
}

export { Greeting };
